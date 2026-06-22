"""US-018 — Out-of-band alerting + harvester alert conditions.

The autonomous harvester needs the operator to hear about breakages without
having to be logged in. ``src.scanner.automation.alert_manager`` only writes
files and is wired to the directional-trade alert vocabulary
(``consecutive_losses`` / ``win_rate_drop``) which makes no sense for a
long-only equity-beta harvester. This module is the equity-side replacement.

Surfaces
--------

* :class:`Notification` — frozen dataclass for the payload that goes out
  the wire. Bytewise stable so the FileNotificationChannel artifact and
  the email body / Slack post all agree on what was sent.
* :class:`NotificationChannel` — protocol that any real out-of-band
  channel implements (``send(notification) -> None``).
* :class:`FileNotificationChannel` — writes each notification as an
  atomic JSON file into an "outbox" directory. This is the channel the
  test suite drives because it requires zero network. Real operators
  watch this directory with ``fswatch``/``inotifywait`` and emit to
  their own out-of-band system.
* :class:`EmailNotificationChannel` — real :mod:`smtplib` SMTP send.
  ``audit_dir`` writes a sidecar copy of every email body so the test
  suite can read it without binding a port.
* :class:`SlackWebhookChannel` — real :mod:`urllib.request` POST to a
  webhook URL. ``audit_dir`` writes a sidecar copy for the same reason.
* :class:`HarvesterAlertManager` — armed once for a universe; fans
  every breach out to all configured channels; debounces with a
  per-alert-type cooldown persisted atomically; refuses to arm without
  a passing ship gate.

Harvester alert conditions
--------------------------

* ``HALT`` — the operator-controlled halt switch is engaged.
* ``BROKER_DISCONNECT`` — broker connectivity lost.
* ``DD_BREACH`` — realized drawdown breached the gate's max-DD threshold.
* ``REBALANCE_FAILED`` — the US-007 rebalance cycle errored out.
* ``STALE_DATA`` — price / EOD data freshness exceeded the threshold.
* ``POSITION_DRIFT`` — actual book drifted from target weights past the
  drift-tolerance band.

The classic directional-trade conditions (``consecutive_losses``,
``win_rate_drop``) are not exposed here — they make no sense for a
long-only systematic harvester.

Ship-gate guard
---------------

:func:`enforce_ship_gate` is byte-for-byte identical to the other
equity guards (rebalance / order_lifecycle / kill_switch /
depth_pricing / executors / shadow_fills) so a fixture that passes one
surface passes them all. Constructing the manager without a passing
gate raises :class:`AlertError`; the gate is re-checked at the start
of every breach check so a gate invalidated mid-run fails closed at
the next call.

Atomic writes and bare-except policy
------------------------------------

Every persisted state mutation uses ``tempfile.mkstemp`` +
``os.replace`` so a crash mid-write leaves the prior valid file on
disk. Zero bare ``except:`` clauses — every catch names its exception
types and either re-raises or logs and surfaces.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"

ALERT_HALT = "HALT"
ALERT_BROKER_DISCONNECT = "BROKER_DISCONNECT"
ALERT_DD_BREACH = "DD_BREACH"
ALERT_REBALANCE_FAILED = "REBALANCE_FAILED"
ALERT_STALE_DATA = "STALE_DATA"
ALERT_POSITION_DRIFT = "POSITION_DRIFT"

HARVESTER_ALERT_TYPES = frozenset(
    {
        ALERT_HALT,
        ALERT_BROKER_DISCONNECT,
        ALERT_DD_BREACH,
        ALERT_REBALANCE_FAILED,
        ALERT_STALE_DATA,
        ALERT_POSITION_DRIFT,
    }
)

SEVERITY_WARNING = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"

DEFAULT_COOLDOWN_SECONDS = 900  # 15 minutes between repeats of the same alert


class AlertError(RuntimeError):
    """Raised on ship-gate refusal, malformed input, or contract bugs.

    Distinct exception type so the autonomous loop (US-017) and the
    rebalance scheduler can match ``except AlertError`` specifically
    without confusing it with channel-side errors.
    """


class ChannelError(RuntimeError):
    """Raised by a channel when the out-of-band send itself fails.

    Separate from :class:`AlertError` so the manager can record per-channel
    failures without treating them as a contract violation that should
    take down the whole alerting subsystem.
    """


# ---------------------------------------------------------------------------
# Ship-gate guard — byte-for-byte identical to the other equity guards.
# ---------------------------------------------------------------------------


def enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse unless ``SHIP_GATE.json`` clears for THIS universe.

    Fail-closed on:

    * file missing
    * file unreadable / not valid JSON
    * payload not a JSON object
    * ``gate_pass`` not exactly ``True``
    * ``universe_hash`` missing / blank / not the expected hash

    Raises:
        AlertError: On any of the above. Identical contract to every
            other equity ship-gate guard.
    """
    path = Path(path)
    if not path.exists():
        raise AlertError(
            f"alerts refuses: ship gate not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AlertError(
            f"alerts refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise AlertError(
            f"alerts refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise AlertError(
            "alerts refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to arm the alert manager)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise AlertError(
            f"alerts refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise AlertError(
            "alerts refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"expected={expected_universe_hash!r}) at {path}"
        )


# ---------------------------------------------------------------------------
# Notification dataclass + channels.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Notification:
    """A single out-of-band notification payload.

    Bytewise stable: serializing the same notification twice produces
    the same JSON. ``timestamp`` is an ISO-8601 UTC string supplied by
    the caller (the manager stamps it from ``datetime.now(timezone.utc)``)
    so deterministic tests can pin the value.
    """

    alert_type: str
    severity: str
    message: str
    timestamp: str
    universe_hash: str
    payload: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def subject(self) -> str:
        return f"[{self.severity}] equity-harvester: {self.alert_type}"


@runtime_checkable
class NotificationChannel(Protocol):
    """Any object with ``send(notification) -> None``.

    Implementations must be side-effect-only and idempotent across
    retries (the manager calls them once per fired alert; the operator
    may replay an outbox file). ``send`` raises :class:`ChannelError`
    on a real send failure so the manager can record per-channel
    failure without aborting the rest of the fan-out.
    """

    def send(self, notification: Notification) -> None:
        ...


def _atomic_write_json(payload: Mapping[str, object], path: Path) -> None:
    """Write ``payload`` to ``path`` via ``mkstemp`` + ``os.replace``.

    Mirrors the other equity surfaces exactly so a crash mid-write has
    identical recovery semantics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(dict(payload), fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.error(
            "atomic write failed for %s: %s", path, exc, exc_info=True
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            logger.warning(
                "failed to remove temp file %s: %s", tmp_path, cleanup_exc
            )
        raise


def _safe_filename_fragment(text: str) -> str:
    """Return a filesystem-safe slug for use in notification filenames."""
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep) or "alert"


class FileNotificationChannel:
    """Writes each notification as an atomic JSON file to an outbox dir.

    This is the real, fully-supported notification channel for operators
    who prefer to wire their own out-of-band relay (``fswatch`` →
    ``osascript`` on macOS, ``inotifywait`` → ``curl`` on Linux, etc.).
    Tests drive this channel because it requires zero network.

    Each ``send`` produces one file at
    ``<outbox>/{timestamp}-{alert_type}-{counter}.json`` with the full
    :class:`Notification` payload. ``counter`` makes the filename unique
    even when several notifications land in the same second.
    """

    def __init__(self, outbox_dir: Path) -> None:
        self.outbox_dir = Path(outbox_dir)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def send(self, notification: Notification) -> None:
        if not isinstance(notification, Notification):
            raise ChannelError(
                "FileNotificationChannel.send requires Notification, got "
                f"{type(notification).__name__}"
            )
        self._counter += 1
        stamp = _safe_filename_fragment(notification.timestamp)
        alert_slug = _safe_filename_fragment(notification.alert_type)
        name = f"{stamp}-{alert_slug}-{self._counter:04d}.json"
        path = self.outbox_dir / name
        try:
            _atomic_write_json(notification.to_dict(), path)
        except OSError as exc:
            raise ChannelError(
                f"FileNotificationChannel failed to write {path}: {exc}"
            ) from exc


@dataclass(frozen=True)
class EmailConfig:
    """SMTP wiring for :class:`EmailNotificationChannel`.

    ``dry_run=True`` skips the live SMTP call entirely and only writes
    the audit sidecar — used in tests and in the operator's pre-flight
    smoke before a real outage. ``audit_dir`` (when set) is a real
    on-disk record of every email body, useful for verification even
    when ``dry_run`` is False.
    """

    smtp_host: str
    smtp_port: int
    from_addr: str
    to_addrs: List[str]
    username: Optional[str] = None
    password: Optional[str] = None
    use_starttls: bool = True
    timeout_seconds: float = 10.0
    dry_run: bool = False
    audit_dir: Optional[Path] = None


class EmailNotificationChannel:
    """Real SMTP send via :mod:`smtplib`.

    Writes a sidecar audit JSON in ``config.audit_dir`` (if set)
    BEFORE doing the SMTP call so a crash mid-send still leaves a
    trace. The SMTP call itself uses ``starttls`` when configured
    and times out at ``config.timeout_seconds``.
    """

    def __init__(self, config: EmailConfig) -> None:
        if not isinstance(config, EmailConfig):
            raise AlertError(
                "EmailNotificationChannel requires EmailConfig, got "
                f"{type(config).__name__}"
            )
        if not config.to_addrs:
            raise AlertError(
                "EmailNotificationChannel requires at least one recipient"
            )
        self.config = config
        self._counter = 0

    def _write_audit(self, notification: Notification, body: str) -> None:
        if self.config.audit_dir is None:
            return
        self._counter += 1
        stamp = _safe_filename_fragment(notification.timestamp)
        alert_slug = _safe_filename_fragment(notification.alert_type)
        name = f"{stamp}-{alert_slug}-{self._counter:04d}.email.json"
        record = {
            "notification": notification.to_dict(),
            "from": self.config.from_addr,
            "to": list(self.config.to_addrs),
            "subject": notification.subject(),
            "body": body,
            "dry_run": bool(self.config.dry_run),
        }
        try:
            _atomic_write_json(record, Path(self.config.audit_dir) / name)
        except OSError as exc:
            raise ChannelError(
                f"EmailNotificationChannel audit write failed: {exc}"
            ) from exc

    def _build_body(self, notification: Notification) -> str:
        lines = [
            f"alert_type:    {notification.alert_type}",
            f"severity:      {notification.severity}",
            f"timestamp:     {notification.timestamp}",
            f"universe_hash: {notification.universe_hash}",
            "",
            notification.message,
            "",
            "payload:",
            json.dumps(notification.payload, indent=2, sort_keys=True, default=str),
        ]
        return "\n".join(lines)

    def send(self, notification: Notification) -> None:
        if not isinstance(notification, Notification):
            raise ChannelError(
                "EmailNotificationChannel.send requires Notification, got "
                f"{type(notification).__name__}"
            )
        body = self._build_body(notification)
        # Always write audit FIRST so a crashed SMTP still leaves a trace.
        self._write_audit(notification, body)

        if self.config.dry_run:
            return

        msg = EmailMessage()
        msg["Subject"] = notification.subject()
        msg["From"] = self.config.from_addr
        msg["To"] = ", ".join(self.config.to_addrs)
        msg.set_content(body)

        try:
            with smtplib.SMTP(
                self.config.smtp_host,
                self.config.smtp_port,
                timeout=self.config.timeout_seconds,
            ) as smtp:
                if self.config.use_starttls:
                    smtp.starttls()
                if self.config.username:
                    smtp.login(self.config.username, self.config.password or "")
                smtp.send_message(msg)
        except (smtplib.SMTPException, OSError) as exc:
            raise ChannelError(
                f"EmailNotificationChannel SMTP send failed: {exc}"
            ) from exc


@dataclass(frozen=True)
class SlackWebhookConfig:
    """Wiring for :class:`SlackWebhookChannel`.

    ``dry_run=True`` skips the live HTTP call entirely (only audit
    sidecar is written). ``audit_dir`` records every POST body.
    """

    webhook_url: str
    channel: Optional[str] = None
    username: str = "equity-harvester"
    icon_emoji: str = ":warning:"
    timeout_seconds: float = 10.0
    dry_run: bool = False
    audit_dir: Optional[Path] = None


class SlackWebhookChannel:
    """Real :mod:`urllib.request` POST to a Slack incoming-webhook URL.

    Slack's incoming-webhook API accepts the standard Block Kit JSON;
    we use the simple ``text``/``username``/``icon_emoji`` form which
    is supported by every workspace and renders cleanly in mobile.
    """

    def __init__(self, config: SlackWebhookConfig) -> None:
        if not isinstance(config, SlackWebhookConfig):
            raise AlertError(
                "SlackWebhookChannel requires SlackWebhookConfig, got "
                f"{type(config).__name__}"
            )
        if not config.webhook_url:
            raise AlertError(
                "SlackWebhookChannel requires a non-empty webhook_url"
            )
        self.config = config
        self._counter = 0

    def _build_payload(self, notification: Notification) -> Dict[str, object]:
        text = (
            f"*{notification.severity}* `{notification.alert_type}`\n"
            f"{notification.message}\n"
            f"`asof={notification.timestamp}` "
            f"`universe_hash={notification.universe_hash}`"
        )
        payload: Dict[str, object] = {
            "text": text,
            "username": self.config.username,
            "icon_emoji": self.config.icon_emoji,
        }
        if self.config.channel:
            payload["channel"] = self.config.channel
        return payload

    def _write_audit(
        self, notification: Notification, slack_payload: Dict[str, object]
    ) -> None:
        if self.config.audit_dir is None:
            return
        self._counter += 1
        stamp = _safe_filename_fragment(notification.timestamp)
        alert_slug = _safe_filename_fragment(notification.alert_type)
        name = f"{stamp}-{alert_slug}-{self._counter:04d}.slack.json"
        record = {
            "notification": notification.to_dict(),
            "slack_payload": slack_payload,
            "webhook_url": self.config.webhook_url,
            "dry_run": bool(self.config.dry_run),
        }
        try:
            _atomic_write_json(record, Path(self.config.audit_dir) / name)
        except OSError as exc:
            raise ChannelError(
                f"SlackWebhookChannel audit write failed: {exc}"
            ) from exc

    def send(self, notification: Notification) -> None:
        if not isinstance(notification, Notification):
            raise ChannelError(
                "SlackWebhookChannel.send requires Notification, got "
                f"{type(notification).__name__}"
            )
        slack_payload = self._build_payload(notification)
        self._write_audit(notification, slack_payload)

        if self.config.dry_run:
            return

        data = json.dumps(slack_payload).encode("utf-8")
        req = urllib.request.Request(
            self.config.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.config.timeout_seconds
            ) as resp:
                status = getattr(resp, "status", 200)
                if status >= 400:
                    raise ChannelError(
                        f"SlackWebhookChannel POST returned HTTP {status}"
                    )
        except urllib.error.URLError as exc:
            raise ChannelError(
                f"SlackWebhookChannel POST failed: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# HarvesterAlertManager — armed alert dispatcher with cooldown + ship gate.
# ---------------------------------------------------------------------------


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HarvesterAlertManager:
    """Harvester alert dispatcher.

    Fans every breach out to every configured channel; debounces with
    a per-alert-type cooldown persisted atomically; refuses to arm
    without a passing ship gate matching the supplied universe hash.

    The check methods (:meth:`check_halt`, :meth:`check_broker_disconnect`,
    :meth:`check_dd_breach`, :meth:`check_rebalance_failed`,
    :meth:`check_stale_data`, :meth:`check_position_drift`) all return
    the :class:`Notification` they emitted (or ``None`` if the condition
    didn't fire or the cooldown blocked it).
    """

    def __init__(
        self,
        channels: List[NotificationChannel],
        ship_gate_path: Path,
        universe_hash: str,
        state_path: Path,
        cooldowns: Optional[Dict[str, float]] = None,
    ) -> None:
        if not isinstance(channels, list) or not channels:
            raise AlertError(
                "HarvesterAlertManager requires at least one channel"
            )
        for ch in channels:
            if not isinstance(ch, NotificationChannel):
                raise AlertError(
                    "HarvesterAlertManager channels must implement "
                    f"NotificationChannel; got {type(ch).__name__}"
                )
        if not isinstance(universe_hash, str) or not universe_hash:
            raise AlertError(
                "HarvesterAlertManager requires a non-empty universe_hash"
            )

        self.channels = list(channels)
        self.ship_gate_path = Path(ship_gate_path)
        self.universe_hash = universe_hash
        self.state_path = Path(state_path)

        # Per-alert-type cooldown in seconds. Unspecified alert types
        # fall back to DEFAULT_COOLDOWN_SECONDS.
        self.cooldowns: Dict[str, float] = {
            alert: float(DEFAULT_COOLDOWN_SECONDS)
            for alert in HARVESTER_ALERT_TYPES
        }
        if cooldowns:
            for k, v in cooldowns.items():
                if k not in HARVESTER_ALERT_TYPES:
                    raise AlertError(
                        f"unknown alert type in cooldowns: {k!r} "
                        f"(known: {sorted(HARVESTER_ALERT_TYPES)})"
                    )
                if not isinstance(v, (int, float)) or v < 0:
                    raise AlertError(
                        f"cooldown for {k!r} must be a non-negative number, "
                        f"got {v!r}"
                    )
                self.cooldowns[k] = float(v)

        enforce_ship_gate(self.ship_gate_path, self.universe_hash)

        self._last_fired: Dict[str, float] = {}
        self._load_state()

    # -- state persistence ------------------------------------------------

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "alert state at %s unreadable (%s); starting fresh",
                self.state_path,
                exc,
            )
            return
        if not isinstance(payload, Mapping):
            logger.warning(
                "alert state at %s is not a JSON object; starting fresh",
                self.state_path,
            )
            return
        last_fired = payload.get("last_fired", {})
        if not isinstance(last_fired, Mapping):
            logger.warning(
                "alert state at %s has invalid last_fired; starting fresh",
                self.state_path,
            )
            return
        for alert_type, ts in last_fired.items():
            if alert_type not in HARVESTER_ALERT_TYPES:
                continue
            try:
                self._last_fired[alert_type] = float(ts)
            except (TypeError, ValueError):
                logger.warning(
                    "alert state at %s has non-numeric ts for %s: %r",
                    self.state_path,
                    alert_type,
                    ts,
                )

    def _save_state(self) -> None:
        payload = {
            "last_fired": dict(self._last_fired),
            "universe_hash": self.universe_hash,
            "last_updated": _now_iso_utc(),
        }
        _atomic_write_json(payload, self.state_path)

    # -- cooldown ---------------------------------------------------------

    def _is_in_cooldown(self, alert_type: str, now: Optional[float] = None) -> bool:
        cooldown = self.cooldowns.get(alert_type, DEFAULT_COOLDOWN_SECONDS)
        if cooldown <= 0:
            return False
        last = self._last_fired.get(alert_type)
        if last is None:
            return False
        now_ts = time.time() if now is None else float(now)
        return (now_ts - last) < cooldown

    # -- core fan-out -----------------------------------------------------

    def notify(
        self,
        alert_type: str,
        severity: str,
        message: str,
        payload: Optional[Dict[str, object]] = None,
        *,
        now: Optional[float] = None,
        force: bool = False,
    ) -> Optional[Notification]:
        """Emit a notification through every configured channel.

        Re-enforces the ship gate before sending so an invalidated gate
        mid-run fails closed at the next breach. Respects the cooldown
        unless ``force=True`` (used by ``check_rebalance_failed`` for
        operator-actionable failures that must page every time).
        """
        if alert_type not in HARVESTER_ALERT_TYPES:
            raise AlertError(
                f"unknown alert type {alert_type!r} "
                f"(known: {sorted(HARVESTER_ALERT_TYPES)})"
            )
        if severity not in {SEVERITY_WARNING, SEVERITY_CRITICAL}:
            raise AlertError(
                f"severity must be {SEVERITY_WARNING!r} or "
                f"{SEVERITY_CRITICAL!r}, got {severity!r}"
            )
        if not isinstance(message, str) or not message.strip():
            raise AlertError("alert message must be a non-empty string")

        # Re-arm: gate may have been invalidated since __init__.
        enforce_ship_gate(self.ship_gate_path, self.universe_hash)

        if not force and self._is_in_cooldown(alert_type, now=now):
            logger.debug(
                "alert %s in cooldown, suppressing: %s",
                alert_type,
                message,
            )
            return None

        notification = Notification(
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=_now_iso_utc(),
            universe_hash=self.universe_hash,
            payload=dict(payload or {}),
        )

        failures: List[str] = []
        for ch in self.channels:
            try:
                ch.send(notification)
            except ChannelError as exc:
                logger.error(
                    "channel %s failed to send %s: %s",
                    type(ch).__name__,
                    alert_type,
                    exc,
                )
                failures.append(f"{type(ch).__name__}: {exc}")
            except (OSError, RuntimeError) as exc:
                logger.error(
                    "channel %s raised unexpected %s on %s: %s",
                    type(ch).__name__,
                    type(exc).__name__,
                    alert_type,
                    exc,
                )
                failures.append(f"{type(ch).__name__}: {exc}")

        # Record the fire even if some channels failed — the cooldown is
        # there to keep us from paging the operator twice for the same
        # condition, not to keep us from retrying every channel forever.
        self._last_fired[alert_type] = (
            time.time() if now is None else float(now)
        )
        self._save_state()

        if failures:
            logger.warning(
                "alert %s emitted with %d/%d channels failing: %s",
                alert_type,
                len(failures),
                len(self.channels),
                "; ".join(failures),
            )
        return notification

    # -- harvester condition checks --------------------------------------

    def check_halt(
        self, halted: bool, reason: str = "", **extras: object
    ) -> Optional[Notification]:
        if not halted:
            return None
        msg = (
            f"equity-harvester is halted: {reason.strip() or 'no reason supplied'}"
        )
        payload: Dict[str, object] = {"reason": reason}
        payload.update(extras)
        return self.notify(ALERT_HALT, SEVERITY_CRITICAL, msg, payload)

    def check_broker_disconnect(
        self,
        connected: bool,
        broker_name: str = "ibkr",
        **extras: object,
    ) -> Optional[Notification]:
        if connected:
            return None
        msg = (
            f"broker {broker_name!r} disconnected; rebalance + kill-switch "
            "calls will fail until reconnect"
        )
        payload: Dict[str, object] = {"broker_name": broker_name}
        payload.update(extras)
        return self.notify(
            ALERT_BROKER_DISCONNECT, SEVERITY_CRITICAL, msg, payload
        )

    def check_dd_breach(
        self,
        current_dd: float,
        threshold: float,
        **extras: object,
    ) -> Optional[Notification]:
        try:
            current_dd_f = float(current_dd)
            threshold_f = float(threshold)
        except (TypeError, ValueError) as exc:
            raise AlertError(
                f"check_dd_breach requires numeric dd/threshold: {exc}"
            ) from exc
        if threshold_f < 0:
            raise AlertError(
                f"check_dd_breach threshold must be >= 0, got {threshold_f}"
            )
        if current_dd_f < threshold_f:
            return None
        msg = (
            f"drawdown {current_dd_f:.3%} breaches threshold {threshold_f:.3%}"
        )
        payload: Dict[str, object] = {
            "current_dd": current_dd_f,
            "threshold": threshold_f,
        }
        payload.update(extras)
        return self.notify(ALERT_DD_BREACH, SEVERITY_CRITICAL, msg, payload)

    def check_rebalance_failed(
        self, error: str, *, force: bool = True, **extras: object
    ) -> Optional[Notification]:
        if not isinstance(error, str) or not error.strip():
            raise AlertError(
                "check_rebalance_failed requires a non-empty error string"
            )
        msg = f"rebalance cycle failed: {error.strip()}"
        payload: Dict[str, object] = {"error": error.strip()}
        payload.update(extras)
        return self.notify(
            ALERT_REBALANCE_FAILED,
            SEVERITY_CRITICAL,
            msg,
            payload,
            force=force,
        )

    def check_stale_data(
        self,
        data_asof: datetime,
        now: datetime,
        threshold_minutes: float,
        **extras: object,
    ) -> Optional[Notification]:
        if not isinstance(data_asof, datetime) or not isinstance(now, datetime):
            raise AlertError(
                "check_stale_data requires datetime objects for "
                "data_asof and now"
            )
        try:
            threshold_minutes_f = float(threshold_minutes)
        except (TypeError, ValueError) as exc:
            raise AlertError(
                f"check_stale_data threshold_minutes must be numeric: {exc}"
            ) from exc
        if threshold_minutes_f <= 0:
            raise AlertError(
                "check_stale_data threshold_minutes must be > 0, "
                f"got {threshold_minutes_f}"
            )

        # Both must be tz-aware (or both naive). Refuse the mixed case
        # rather than silently subtracting them.
        if (data_asof.tzinfo is None) != (now.tzinfo is None):
            raise AlertError(
                "check_stale_data: data_asof and now must both be "
                "tz-aware or both naive"
            )

        age_seconds = (now - data_asof).total_seconds()
        threshold_seconds = threshold_minutes_f * 60.0
        if age_seconds < threshold_seconds:
            return None
        msg = (
            f"data is {age_seconds / 60.0:.1f} min old "
            f"(threshold {threshold_minutes_f:.1f} min); refusing to trade "
            "on stale prices"
        )
        payload: Dict[str, object] = {
            "data_asof": data_asof.isoformat(),
            "now": now.isoformat(),
            "age_seconds": age_seconds,
            "threshold_minutes": threshold_minutes_f,
        }
        payload.update(extras)
        return self.notify(ALERT_STALE_DATA, SEVERITY_CRITICAL, msg, payload)

    def check_position_drift(
        self,
        actual_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
        threshold: float,
        **extras: object,
    ) -> Optional[Notification]:
        if not isinstance(actual_weights, Mapping) or not isinstance(
            target_weights, Mapping
        ):
            raise AlertError(
                "check_position_drift requires mapping actual_weights and "
                "target_weights"
            )
        try:
            threshold_f = float(threshold)
        except (TypeError, ValueError) as exc:
            raise AlertError(
                f"check_position_drift threshold must be numeric: {exc}"
            ) from exc
        if threshold_f < 0:
            raise AlertError(
                f"check_position_drift threshold must be >= 0, got {threshold_f}"
            )

        all_tickers = set(actual_weights) | set(target_weights)
        max_drift = 0.0
        worst_ticker: Optional[str] = None
        per_ticker_drift: Dict[str, float] = {}
        for ticker in all_tickers:
            try:
                a = float(actual_weights.get(ticker, 0.0))
                t = float(target_weights.get(ticker, 0.0))
            except (TypeError, ValueError) as exc:
                raise AlertError(
                    f"check_position_drift: non-numeric weight for "
                    f"{ticker!r}: {exc}"
                ) from exc
            drift = abs(a - t)
            per_ticker_drift[ticker] = drift
            if drift > max_drift:
                max_drift = drift
                worst_ticker = ticker

        if max_drift <= threshold_f:
            return None

        msg = (
            f"position drift {max_drift:.4f} > {threshold_f:.4f} "
            f"on {worst_ticker!r}; book has decoupled from target weights"
        )
        payload: Dict[str, object] = {
            "max_drift": max_drift,
            "threshold": threshold_f,
            "worst_ticker": worst_ticker,
            "per_ticker_drift": per_ticker_drift,
        }
        payload.update(extras)
        return self.notify(
            ALERT_POSITION_DRIFT, SEVERITY_WARNING, msg, payload
        )


__all__ = [
    "ALERT_BROKER_DISCONNECT",
    "ALERT_DD_BREACH",
    "ALERT_HALT",
    "ALERT_POSITION_DRIFT",
    "ALERT_REBALANCE_FAILED",
    "ALERT_STALE_DATA",
    "AlertError",
    "ChannelError",
    "DEFAULT_COOLDOWN_SECONDS",
    "EmailConfig",
    "EmailNotificationChannel",
    "FileNotificationChannel",
    "HARVESTER_ALERT_TYPES",
    "HarvesterAlertManager",
    "Notification",
    "NotificationChannel",
    "SEVERITY_CRITICAL",
    "SEVERITY_WARNING",
    "SHIP_GATE_PATH_DEFAULT",
    "SlackWebhookChannel",
    "SlackWebhookConfig",
    "enforce_ship_gate",
]
