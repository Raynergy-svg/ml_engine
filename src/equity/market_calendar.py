"""US-010 — Equity market-calendar / session / halt gate.

The FX :meth:`ScannerConfig.is_market_open` (``src/scanner/config.py:1420``)
models a 24/5 FX market and is wrong for cash equities. The equity
harvester's rebalance / order path must be gated by the real US equity
session: NYSE/Nasdaq regular hours 09:30-16:00 ET, with holidays AND
half-days handled correctly. This module provides that gate.

Three responsibilities, separated:

1. :class:`EquityMarketCalendar` — wraps ``exchange_calendars`` (``XNYS``
   by default) and answers "is the regular session open at this minute?".
   Holidays and half-days come from the underlying calendar; the half-day
   early close (13:00 ET on the day after Thanksgiving and the half before
   Christmas/Independence Day) is honoured because we read
   ``is_open_on_minute`` rather than hard-coding 09:30-16:00.
2. :class:`HaltRegistry` — per-ticker halt state persisted atomically. A
   broker that surfaces a halt/reject via :class:`HaltError` causes the
   router to defer that ticker until the registry clears it (manual
   resume) OR a TTL elapses (timed resume). Never silently dropped.
3. :class:`EquitySessionGate` — top-level seam used by the rebalance
   executor. Construction is fail-closed: ``create()`` enforces the
   ship-gate guard (``trained_data/backtests/SHIP_GATE.json`` with
   ``gate_pass==True`` AND matching ``universe_hash``) before any method
   is reachable. Provides :meth:`guard_session` (refuse rebalance if
   market closed) and :meth:`safe_send` (per-order halt-aware wrapper).

Hard rules (mirror ``rebalance.py``):

* Real classes + real disk via ``tmp_path`` in tests — no
  ``unittest.mock``.
* Atomic writes: ``tempfile.mkstemp`` + :func:`os.replace`. A crash
  mid-write leaves the prior state on disk, never a half-written one.
* No bare ``except``: every catch names specific exceptions, logs the
  error, and re-raises or surfaces the failure to the caller.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

import pandas as pd

try:
    import exchange_calendars as ec
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "src.equity.market_calendar requires `exchange_calendars` "
        "(pip install exchange_calendars). The equity harvester refuses "
        "to run without a real NYSE/Nasdaq calendar."
    ) from exc

from src.equity.rebalance import Order

logger = logging.getLogger(__name__)


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
DEFAULT_EXCHANGE = "XNYS"  # NYSE; XNAS shares the same regular session
HALT_STATE_VERSION = 1


class MarketCalendarError(RuntimeError):
    """Raised when calendar inputs or persisted state make a run unsafe."""


class HaltError(RuntimeError):
    """Broker-side signal that an order was rejected because of a halt.

    The :class:`EquitySessionGate` catches this around ``send_order``,
    records the halt in :class:`HaltRegistry`, and defers the order.
    Callers should raise this from their broker callback when they get
    an LULD trading-pause / halt / reject-due-to-halt response.
    """


class OrderOutcome(str, Enum):
    """Per-order outcome from :meth:`EquitySessionGate.safe_send`."""

    FILLED = "FILLED"
    DEFERRED_HALT = "DEFERRED_HALT"
    DEFERRED_CLOSED = "DEFERRED_CLOSED"
    REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
class EquityMarketCalendar:
    """Thin wrapper around ``exchange_calendars`` for the equity path.

    Default exchange is ``XNYS`` (New York Stock Exchange). XNYS and XNAS
    share the same 09:30-16:00 ET regular session and the same holiday
    calendar for US equities, so picking either is correct for the
    harvester's blended NYSE/Nasdaq universe.
    """

    def __init__(self, exchange: str = DEFAULT_EXCHANGE) -> None:
        try:
            self._cal = ec.get_calendar(exchange)
        except (ec.errors.InvalidCalendarName, ValueError) as exc:
            raise MarketCalendarError(
                f"unknown exchange calendar {exchange!r}: {exc}"
            ) from exc
        self.exchange = str(exchange)

    @staticmethod
    def _to_utc(now: object) -> pd.Timestamp:
        ts = pd.Timestamp(now)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    def is_session_open(self, now: object) -> bool:
        """``True`` iff ``now`` is inside the regular session.

        Holidays return ``False``; half-day close (e.g. 13:00 ET on the
        day after Thanksgiving) returns ``False`` after the early close.
        """
        return bool(self._cal.is_open_on_minute(self._to_utc(now)))

    def session_status(self, now: object) -> Dict[str, object]:
        """Structured status dict — useful for logging / surfacing.

        Always returns a stable schema: ``{exchange, now_utc, is_open,
        reason_code, message, next_open_utc, next_close_utc}``.
        """
        now_utc = self._to_utc(now)
        is_open = self.is_session_open(now_utc)

        reason_code = "open" if is_open else "closed"
        message: str
        try:
            next_open = pd.Timestamp(self._cal.next_open(now_utc))
            next_close = pd.Timestamp(self._cal.next_close(now_utc))
        except (ValueError, KeyError) as exc:
            logger.warning(
                "calendar lookup failed at %s: %s — surfacing partial status",
                now_utc,
                exc,
            )
            next_open = pd.NaT
            next_close = pd.NaT

        if is_open:
            message = (
                f"{self.exchange} regular session open at "
                f"{now_utc.isoformat()}"
            )
        else:
            message = (
                f"{self.exchange} regular session closed at "
                f"{now_utc.isoformat()}"
            )

        return {
            "exchange": self.exchange,
            "now_utc": now_utc.isoformat(),
            "is_open": bool(is_open),
            "reason_code": reason_code,
            "message": message,
            "next_open_utc": (
                next_open.isoformat() if not pd.isna(next_open) else None
            ),
            "next_close_utc": (
                next_close.isoformat() if not pd.isna(next_close) else None
            ),
        }


# ---------------------------------------------------------------------------
# Halt registry (persisted)
# ---------------------------------------------------------------------------
@dataclass
class HaltRecord:
    """A single halt entry. ``until`` is an optional UTC ISO TTL."""

    ticker: str
    reason: str
    halted_at: str
    until: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ticker": str(self.ticker),
            "reason": str(self.reason),
            "halted_at": str(self.halted_at),
            "until": str(self.until) if self.until else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "HaltRecord":
        until = payload.get("until")
        return cls(
            ticker=str(payload["ticker"]),
            reason=str(payload.get("reason", "")),
            halted_at=str(payload["halted_at"]),
            until=str(until) if until else None,
        )


@dataclass
class HaltState:
    """Persisted halt-registry snapshot."""

    version: int = HALT_STATE_VERSION
    halts: Dict[str, HaltRecord] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": int(self.version),
            "halts": {t: r.to_dict() for t, r in self.halts.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "HaltState":
        raw = payload.get("halts") or {}
        if not isinstance(raw, Mapping):
            raise MarketCalendarError("halts must be a JSON object")
        halts = {
            str(t): HaltRecord.from_dict(rec)
            for t, rec in raw.items()
            if isinstance(rec, Mapping)
        }
        return cls(
            version=int(payload.get("version", HALT_STATE_VERSION)),
            halts=halts,
        )


def _atomic_write_json(payload: dict, path: Path) -> None:
    """Atomic JSON write: ``mkstemp`` sibling + :func:`os.replace`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
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


class HaltRegistry:
    """Per-ticker halt registry persisted atomically to disk.

    A ticker is halted iff a :class:`HaltRecord` exists AND either has no
    ``until`` (manual resume only) or ``now < until`` (TTL still active).
    Expired TTL entries are evicted by :meth:`is_halted` on read so the
    on-disk state stays small.
    """

    def __init__(self, state_path: Path) -> None:
        self.state_path = Path(state_path)

    # ------------------------------------------------------------------
    def load(self) -> HaltState:
        path = self.state_path
        if not path.exists():
            return HaltState()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "cannot read halt state %s: %s — resetting", path, exc
            )
            return HaltState()
        if not isinstance(payload, dict):
            logger.error(
                "halt state %s is not a JSON object — resetting", path
            )
            return HaltState()
        try:
            return HaltState.from_dict(payload)
        except (MarketCalendarError, KeyError, TypeError, ValueError) as exc:
            logger.error(
                "halt state %s is malformed: %s — resetting", path, exc
            )
            return HaltState()

    def save(self, state: HaltState) -> None:
        _atomic_write_json(state.to_dict(), self.state_path)

    # ------------------------------------------------------------------
    def register_halt(
        self,
        ticker: str,
        *,
        now: object,
        reason: str = "",
        until: Optional[object] = None,
    ) -> HaltRecord:
        """Record a halt for ``ticker``; overwrites any prior record."""
        if not isinstance(ticker, str) or not ticker:
            raise MarketCalendarError("ticker must be a non-empty string")
        state = self.load()
        halted_at = EquityMarketCalendar._to_utc(now).isoformat()
        until_iso: Optional[str]
        if until is None:
            until_iso = None
        else:
            until_iso = EquityMarketCalendar._to_utc(until).isoformat()
        rec = HaltRecord(
            ticker=ticker,
            reason=str(reason),
            halted_at=halted_at,
            until=until_iso,
        )
        state.halts[ticker] = rec
        self.save(state)
        return rec

    def clear_halt(self, ticker: str) -> bool:
        """Remove ``ticker`` from the registry. Returns ``True`` if it was
        present.
        """
        state = self.load()
        if ticker not in state.halts:
            return False
        del state.halts[ticker]
        self.save(state)
        return True

    def is_halted(self, ticker: str, *, now: object) -> bool:
        """``True`` iff a non-expired halt record exists for ``ticker``."""
        state = self.load()
        rec = state.halts.get(ticker)
        if rec is None:
            return False
        if rec.until is None:
            return True
        try:
            until_ts = EquityMarketCalendar._to_utc(rec.until)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "halt record for %s has bad until %r: %s — treating halted",
                ticker,
                rec.until,
                exc,
            )
            return True
        now_ts = EquityMarketCalendar._to_utc(now)
        if now_ts >= until_ts:
            # TTL elapsed: evict and clear.
            del state.halts[ticker]
            self.save(state)
            return False
        return True

    def list_active(self, *, now: object) -> List[HaltRecord]:
        """Return all currently-halted records (TTL filter applied)."""
        state = self.load()
        active: List[HaltRecord] = []
        for ticker, rec in list(state.halts.items()):
            if self.is_halted(ticker, now=now):
                active.append(rec)
        return active


# ---------------------------------------------------------------------------
# Ship-gate guard (mirrors src.equity.rebalance._enforce_ship_gate)
# ---------------------------------------------------------------------------
def _enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse to run unless the ship gate clears for THIS universe.

    Fail-closed: missing file, malformed JSON, missing keys, ``gate_pass``
    not exactly ``True``, hash mismatch all raise
    :class:`MarketCalendarError`.
    """
    path = Path(path)
    if not path.exists():
        raise MarketCalendarError(
            f"ship gate refuses: file not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarketCalendarError(
            f"ship gate refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise MarketCalendarError(
            f"ship gate refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise MarketCalendarError(
            "ship gate refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to unblock the equity session gate)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise MarketCalendarError(
            f"ship gate refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise MarketCalendarError(
            "ship gate refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"snapshot={expected_universe_hash!r}) at {path}"
        )


# ---------------------------------------------------------------------------
# Session gate (top-level seam)
# ---------------------------------------------------------------------------
@dataclass
class SafeSendResult:
    """Outcome of one :meth:`EquitySessionGate.safe_send` call."""

    outcome: str  # OrderOutcome value
    order: Order
    reason: str = ""


@dataclass
class EquitySessionGate:
    """Ship-gate-guarded, halt-aware order seam for the equity path.

    Construction is via :meth:`create` so the ship-gate guard runs once,
    deterministically, before any other method is reachable. Mirrors the
    :meth:`RebalanceScheduler.create` pattern from US-007.
    """

    universe_hash: str
    ship_gate_path: Path
    halt_registry: HaltRegistry
    calendar: EquityMarketCalendar

    @classmethod
    def create(
        cls,
        *,
        universe_hash: str,
        ship_gate_path: Path,
        halt_state_path: Path,
        exchange: str = DEFAULT_EXCHANGE,
    ) -> "EquitySessionGate":
        """Run ship-gate guard then return a fresh session gate."""
        if not isinstance(universe_hash, str) or not universe_hash:
            raise MarketCalendarError(
                "universe_hash must be a non-empty string"
            )
        _enforce_ship_gate(Path(ship_gate_path), universe_hash)
        return cls(
            universe_hash=str(universe_hash),
            ship_gate_path=Path(ship_gate_path),
            halt_registry=HaltRegistry(Path(halt_state_path)),
            calendar=EquityMarketCalendar(exchange=exchange),
        )

    # ------------------------------------------------------------------
    def guard_session(self, *, now: object) -> None:
        """Refuse to rebalance unless the regular session is open.

        Raises :class:`MarketCalendarError` outside the session so the
        caller cannot accidentally fire a rebalance into a closed
        market. Holidays, weekends, early-close half-days, and
        pre-/post-market all hit this branch.
        """
        if not self.calendar.is_session_open(now):
            status = self.calendar.session_status(now)
            raise MarketCalendarError(
                f"refuse rebalance: {status['message']} "
                f"(next_open={status['next_open_utc']})"
            )

    # ------------------------------------------------------------------
    def safe_send(
        self,
        order: Order,
        send_order: Callable[[Order], bool],
        *,
        now: object,
        halt_ttl: Optional[object] = None,
    ) -> SafeSendResult:
        """Per-order halt- and session-aware wrapper around ``send_order``.

        Behaviour:

        * Session closed → :attr:`OrderOutcome.DEFERRED_CLOSED` (order
          left untouched; caller retries on the next open).
        * Ticker already halted → :attr:`OrderOutcome.DEFERRED_HALT`
          (no broker call; not silently dropped).
        * ``send_order`` raises :class:`HaltError` → halt is recorded in
          the registry with ``halt_ttl`` (defaults to None = manual
          resume) and the order is deferred.
        * ``send_order`` returns ``True`` → :attr:`OrderOutcome.FILLED`.
        * ``send_order`` returns ``False`` →
          :attr:`OrderOutcome.REJECTED` (graceful reject, not a halt).

        Any other exception from ``send_order`` is logged and re-raised
        so the caller's halt/retry logic stays in control. The order is
        never silently dropped.
        """
        if not self.calendar.is_session_open(now):
            status = self.calendar.session_status(now)
            logger.info(
                "safe_send defer: market closed (%s) — order %s",
                status["message"],
                order.client_order_id,
            )
            return SafeSendResult(
                outcome=OrderOutcome.DEFERRED_CLOSED.value,
                order=order,
                reason=str(status["message"]),
            )

        if self.halt_registry.is_halted(order.ticker, now=now):
            logger.info(
                "safe_send defer: ticker %s halted — order %s",
                order.ticker,
                order.client_order_id,
            )
            return SafeSendResult(
                outcome=OrderOutcome.DEFERRED_HALT.value,
                order=order,
                reason=f"ticker {order.ticker} halted in registry",
            )

        try:
            ok = bool(send_order(order))
        except HaltError as exc:
            reason = str(exc) or "broker raised HaltError"
            self.halt_registry.register_halt(
                order.ticker,
                now=now,
                reason=reason,
                until=halt_ttl,
            )
            logger.warning(
                "safe_send: %s halted by broker (%s) — order %s deferred",
                order.ticker,
                reason,
                order.client_order_id,
            )
            return SafeSendResult(
                outcome=OrderOutcome.DEFERRED_HALT.value,
                order=order,
                reason=reason,
            )
        except Exception as exc:
            logger.error(
                "safe_send: send_order raised non-halt exception for %s: %s",
                order.client_order_id,
                exc,
                exc_info=True,
            )
            raise

        if ok:
            return SafeSendResult(
                outcome=OrderOutcome.FILLED.value,
                order=order,
                reason="broker fill",
            )
        return SafeSendResult(
            outcome=OrderOutcome.REJECTED.value,
            order=order,
            reason="broker returned False",
        )

    # ------------------------------------------------------------------
    def deferred_tickers(self, *, now: object) -> List[str]:
        """Currently-halted tickers (TTL filter applied)."""
        return [r.ticker for r in self.halt_registry.list_active(now=now)]


__all__ = [
    "DEFAULT_EXCHANGE",
    "EquityMarketCalendar",
    "EquitySessionGate",
    "HALT_STATE_VERSION",
    "HaltError",
    "HaltRecord",
    "HaltRegistry",
    "HaltState",
    "MarketCalendarError",
    "OrderOutcome",
    "SHIP_GATE_PATH_DEFAULT",
    "SafeSendResult",
]
