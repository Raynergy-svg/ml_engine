"""US-007 — Rebalance scheduler with idempotent restart.

The harvester emits a *target* book; this module is the seam between that
target and the broker. It owns three responsibilities and only three:

1. **Triggering** — fire a rebalance on a configurable calendar
   (``rebalance_freq_days``, default 30) OR when any name's drift between
   current actual and target exceeds the no-trade band.
2. **Order delta** — translate ``(current, target)`` into per-name BUY/SELL
   :class:`Order` objects with stable, deterministic
   :attr:`Order.client_order_id` so the broker (or any restart of this
   module) can dedupe them by ID.
3. **Idempotent restart** — every state mutation (plan creation, per-order
   send + fill, plan finalisation) is persisted atomically to disk. On
   restart, :meth:`RebalanceScheduler.reconcile_and_resume` loads the
   active plan with its prior order statuses preserved; the executor then
   sends only the orders still in non-FILLED state. AC: "kill after 2 of 5
   orders → restart sends exactly the missing 3, no double-fill".

Ship-gate guard
---------------

Construction is fail-closed: :func:`RebalanceScheduler.create` reads
``trained_data/backtests/SHIP_GATE.json`` and refuses unless
``gate_pass==True`` AND the file's ``universe_hash`` matches the snapshot
the scheduler is being attached to. Missing file, malformed JSON, false
gate, hash mismatch — all raise :class:`RebalanceError`.

Atomic writes
-------------

State persistence uses the same ``tempfile.mkstemp`` + :func:`os.replace`
pattern as :mod:`src.equity.harvester_strategy`: a crash mid-write leaves
the prior state on disk, never a half-written one.

No mocks; tests drive real classes against real ``tmp_path`` disk state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
# Canonical on-disk location for the rebalance scheduler's persisted state
# (last_rebalance_asof, current_actual_weights, active_plan with per-order
# PENDING/SENT/FILLED/FAILED status). The scheduler accepts ``state_path`` at
# construction; this constant is the single source of truth the loop launcher
# AND the TUI reader share so the wiring can never drift to a hardcoded string.
STATE_PATH_DEFAULT = "trained_data/equity/rebalance_state.json"
STATE_VERSION = 1
DEFAULT_NO_TRADE_BAND = 0.005  # 0.5% per-name absolute weight band
DEFAULT_REBALANCE_FREQ_DAYS = 30


class RebalanceError(RuntimeError):
    """Raised when scheduler inputs or persisted state make a run unsafe."""


class OrderStatus(str, Enum):
    """Lifecycle states for an :class:`Order`."""

    PENDING = "PENDING"
    SENT = "SENT"
    FILLED = "FILLED"
    FAILED = "FAILED"


class ExecResult(str, Enum):
    """Outcome an :meth:`RebalanceScheduler.execute_plan` ``send_order``
    callback reports for a single order.

    The split exists to stop conflating a CONFIRMED fill with mere broker
    ACCEPTANCE. Booking acceptance as a fill corrupted the actual-weights
    ledger that drives the next drift calc — see
    ``docs/equity-execution-review-2026-06-21.md`` (C3) and the 2026-06-30
    re-review. ``ACCEPTED`` is fail-closed: the order is marked SENT
    (in-flight) and is NOT counted as a position until a reconcile pass
    confirms real broker shares.
    """

    FILLED = "FILLED"      # broker confirmed shares executed
    ACCEPTED = "ACCEPTED"  # broker accepted the order; fill NOT yet confirmed
    REJECTED = "REJECTED"  # broker rejected / no order placed


def normalize_exec_result(raw: "ExecResult | bool | None") -> "ExecResult":
    """Map a ``send_order`` return value onto an :class:`ExecResult`.

    Backward-compatible with the legacy ``bool`` contract: ``True`` is a
    CONFIRMED fill (the shadow simulator fills synchronously), ``False`` is a
    rejection. A broker that merely ACCEPTED an order without confirming a
    fill MUST return :attr:`ExecResult.ACCEPTED` — never bare ``True`` — or it
    re-introduces the C3 phantom-fill bug.
    """
    if isinstance(raw, ExecResult):
        return raw
    return ExecResult.FILLED if bool(raw) else ExecResult.REJECTED


@dataclass
class Order:
    """A single BUY/SELL order keyed by a stable ``client_order_id``.

    ``client_order_id`` is a deterministic hash of
    ``(rebalance_id, ticker, side, target_weight)`` — the same inputs on
    restart produce the same ID, so the broker can dedupe and we never
    double-fill.
    """

    client_order_id: str
    ticker: str
    side: str  # "BUY" or "SELL"
    weight_delta: float
    target_weight: float
    status: str = OrderStatus.PENDING.value
    fill_weight: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Order":
        return cls(
            client_order_id=str(payload["client_order_id"]),
            ticker=str(payload["ticker"]),
            side=str(payload["side"]),
            weight_delta=float(payload["weight_delta"]),
            target_weight=float(payload["target_weight"]),
            status=str(payload.get("status", OrderStatus.PENDING.value)),
            fill_weight=float(payload.get("fill_weight", 0.0)),
        )

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED.value


@dataclass
class RebalancePlan:
    """A single rebalance cycle: target + the orders that take us there."""

    rebalance_id: str
    asof: str
    target_weights: Dict[str, float]
    orders: List[Order]
    no_trade_band: float

    def remaining_orders(self) -> List[Order]:
        return [o for o in self.orders if not o.is_filled]

    def filled_orders(self) -> List[Order]:
        return [o for o in self.orders if o.is_filled]

    def to_dict(self) -> dict:
        return {
            "rebalance_id": self.rebalance_id,
            "asof": self.asof,
            "target_weights": dict(self.target_weights),
            "orders": [o.to_dict() for o in self.orders],
            "no_trade_band": float(self.no_trade_band),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RebalancePlan":
        target = payload.get("target_weights") or {}
        orders_payload = payload.get("orders") or []
        if not isinstance(target, Mapping):
            raise RebalanceError("plan.target_weights must be a mapping")
        if not isinstance(orders_payload, list):
            raise RebalanceError("plan.orders must be a list")
        return cls(
            rebalance_id=str(payload["rebalance_id"]),
            asof=str(payload["asof"]),
            target_weights={str(k): float(v) for k, v in target.items()},
            orders=[Order.from_dict(o) for o in orders_payload],
            no_trade_band=float(payload.get("no_trade_band", 0.0)),
        )


@dataclass
class RebalanceState:
    """Persisted scheduler state — survives restart.

    Holds the last-finalised rebalance asof, the current actual weights
    (updated as each fill lands), and the in-flight :class:`RebalancePlan`
    (serialised as a dict so we don't need a custom JSON encoder).
    """

    version: int = STATE_VERSION
    last_rebalance_asof: Optional[str] = None
    current_actual_weights: Dict[str, float] = field(default_factory=dict)
    active_plan: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "version": int(self.version),
            "last_rebalance_asof": self.last_rebalance_asof,
            "current_actual_weights": {
                str(k): float(v) for k, v in self.current_actual_weights.items()
            },
            "active_plan": self.active_plan,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RebalanceState":
        last = payload.get("last_rebalance_asof")
        cur = payload.get("current_actual_weights") or {}
        if not isinstance(cur, Mapping):
            raise RebalanceError(
                "state.current_actual_weights must be a mapping"
            )
        active = payload.get("active_plan")
        if active is not None and not isinstance(active, Mapping):
            raise RebalanceError("state.active_plan must be a mapping or null")
        return cls(
            version=int(payload.get("version", STATE_VERSION)),
            last_rebalance_asof=str(last) if last else None,
            current_actual_weights={
                str(k): float(v) for k, v in cur.items()
            },
            active_plan=dict(active) if active is not None else None,
        )


def _atomic_write_json(payload: dict, path: Path) -> None:
    """Write ``payload`` to ``path`` via ``mkstemp`` + :func:`os.replace`."""
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


def _enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse to run unless the ship gate clears for THIS universe.

    Fail-closed: missing file, malformed JSON, missing keys, ``gate_pass``
    not exactly ``True``, hash mismatch all raise :class:`RebalanceError`.
    """
    path = Path(path)
    if not path.exists():
        raise RebalanceError(
            f"ship gate refuses: file not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RebalanceError(
            f"ship gate refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RebalanceError(
            f"ship gate refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise RebalanceError(
            "ship gate refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to unblock the scheduler)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise RebalanceError(
            f"ship gate refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise RebalanceError(
            "ship gate refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"snapshot={expected_universe_hash!r}) at {path}"
        )


def _make_rebalance_id(asof: str, universe_hash: str) -> str:
    """Deterministic per-cycle ID.

    Same ``(asof, universe_hash)`` always produces the same ID — so a
    restart that recomputes the plan from the same target gets the same
    ``client_order_id``s downstream.
    """
    short_hash = universe_hash[:8] if universe_hash else "00000000"
    safe_asof = asof.replace(":", "").replace("-", "").replace(" ", "T")
    return f"RB-{safe_asof}-{short_hash}"


def _client_order_id(
    rebalance_id: str,
    ticker: str,
    side: str,
    target_weight: float,
) -> str:
    """Stable deterministic ``client_order_id``.

    Hash of ``(rebalance_id, ticker, side, target_weight)`` → 16-hex prefix.
    Same inputs on restart → same ID → broker dedupe → no double-fill.
    """
    key = f"{rebalance_id}|{ticker}|{side}|{target_weight:.10f}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{rebalance_id}-{ticker}-{digest}"


def _asof_iso(asof: object) -> str:
    """Coerce ``asof`` to a UTC ISO string."""
    ts = pd.Timestamp(asof)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def plan_rebalance(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    *,
    asof: str,
    universe_hash: str,
    no_trade_band: float = DEFAULT_NO_TRADE_BAND,
) -> RebalancePlan:
    """Build a :class:`RebalancePlan` from current vs target.

    A ticker is included only if ``|target - current| > no_trade_band``.
    Closures (``target == 0`` while ``current != 0``) are unconditional —
    we always close a position regardless of the band so a stale name
    doesn't linger.
    """
    if no_trade_band < 0:
        raise RebalanceError(
            f"no_trade_band must be >= 0, got {no_trade_band!r}"
        )

    rb_id = _make_rebalance_id(asof, universe_hash)
    tickers = sorted(set(current_weights) | set(target_weights))
    orders: List[Order] = []
    target_clean: Dict[str, float] = {}
    for ticker in tickers:
        cur = float(current_weights.get(ticker, 0.0))
        tgt = float(target_weights.get(ticker, 0.0))
        target_clean[ticker] = tgt
        delta = tgt - cur
        is_closure = (tgt == 0.0) and (cur != 0.0)
        if not is_closure and abs(delta) <= no_trade_band:
            continue
        if delta == 0.0:
            continue
        side = "BUY" if delta > 0 else "SELL"
        orders.append(
            Order(
                client_order_id=_client_order_id(rb_id, ticker, side, tgt),
                ticker=ticker,
                side=side,
                weight_delta=float(delta),
                target_weight=float(tgt),
            )
        )
    return RebalancePlan(
        rebalance_id=rb_id,
        asof=asof,
        target_weights=target_clean,
        orders=orders,
        no_trade_band=float(no_trade_band),
    )


def _drift_exceeds_band(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    no_trade_band: float,
) -> bool:
    """``True`` if any name's drift exceeds the band."""
    tickers = set(current_weights) | set(target_weights)
    for t in tickers:
        cur = float(current_weights.get(t, 0.0))
        tgt = float(target_weights.get(t, 0.0))
        if abs(tgt - cur) > no_trade_band:
            return True
    return False


@dataclass
class RebalanceScheduler:
    """Idempotent rebalance scheduler.

    Construction goes through :meth:`create` so the ship-gate guard runs
    once, deterministically, before any other method is reachable.
    """

    universe_hash: str
    ship_gate_path: Path
    state_path: Path
    no_trade_band: float = DEFAULT_NO_TRADE_BAND
    rebalance_freq_days: int = DEFAULT_REBALANCE_FREQ_DAYS

    @classmethod
    def create(
        cls,
        *,
        universe_hash: str,
        ship_gate_path: Path,
        state_path: Path,
        no_trade_band: float = DEFAULT_NO_TRADE_BAND,
        rebalance_freq_days: int = DEFAULT_REBALANCE_FREQ_DAYS,
    ) -> "RebalanceScheduler":
        """Run ship-gate guard then return a fresh scheduler instance."""
        if not isinstance(universe_hash, str) or not universe_hash:
            raise RebalanceError(
                "universe_hash must be a non-empty string"
            )
        if no_trade_band < 0:
            raise RebalanceError(
                f"no_trade_band must be >= 0, got {no_trade_band!r}"
            )
        if rebalance_freq_days < 1:
            raise RebalanceError(
                "rebalance_freq_days must be >= 1, "
                f"got {rebalance_freq_days!r}"
            )
        _enforce_ship_gate(Path(ship_gate_path), universe_hash)
        return cls(
            universe_hash=str(universe_hash),
            ship_gate_path=Path(ship_gate_path),
            state_path=Path(state_path),
            no_trade_band=float(no_trade_band),
            rebalance_freq_days=int(rebalance_freq_days),
        )

    # ------------------------------------------------------------------
    # State I/O
    # ------------------------------------------------------------------
    def load_state(self) -> RebalanceState:
        """Read persisted state from disk; return a fresh state if absent.

        A corrupt file is treated like a missing one (logged, reset). The
        scheduler is restart-safe by design — a crash mid-write leaves the
        prior file on disk, and a fresh state is recoverable.
        """
        path = Path(self.state_path)
        if not path.exists():
            return RebalanceState()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "cannot read state %s: %s — resetting to fresh state",
                path,
                exc,
            )
            return RebalanceState()
        if not isinstance(payload, dict):
            logger.error(
                "state file %s is not a JSON object — resetting", path
            )
            return RebalanceState()
        try:
            return RebalanceState.from_dict(payload)
        except RebalanceError as exc:
            logger.error(
                "state file %s is malformed: %s — resetting", path, exc
            )
            return RebalanceState()

    def write_state(self, state: RebalanceState) -> None:
        """Atomically persist ``state`` to ``state_path``."""
        _atomic_write_json(state.to_dict(), Path(self.state_path))

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------
    def should_rebalance(
        self,
        *,
        now: pd.Timestamp,
        current_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
        last_rebalance_asof: Optional[str],
    ) -> Tuple[bool, str]:
        """Return ``(fire, reason)``.

        Reasons: ``"initial"`` (first run), ``"drift"`` (any name beyond
        the band), ``"calendar"`` (>= ``rebalance_freq_days`` since last
        finalised rebalance), or ``"none"``.
        """
        if last_rebalance_asof is None:
            return True, "initial"
        if _drift_exceeds_band(
            current_weights, target_weights, self.no_trade_band
        ):
            return True, "drift"
        try:
            last_ts = pd.Timestamp(last_rebalance_asof)
            if last_ts.tz is None:
                last_ts = last_ts.tz_localize("UTC")
            now_ts = pd.Timestamp(now)
            if now_ts.tz is None:
                now_ts = now_ts.tz_localize("UTC")
            else:
                now_ts = now_ts.tz_convert("UTC")
            days = (now_ts - last_ts).days
        except (ValueError, TypeError) as exc:
            logger.warning(
                "bad last_rebalance_asof %r: %s — firing calendar trigger",
                last_rebalance_asof,
                exc,
            )
            return True, "calendar"
        if days >= self.rebalance_freq_days:
            return True, "calendar"
        return False, "none"

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    def plan_and_persist(
        self,
        *,
        target_weights: Mapping[str, float],
        asof: pd.Timestamp,
    ) -> RebalancePlan:
        """Build a fresh plan from current_actual → target; persist it.

        Refuses if an :attr:`RebalanceState.active_plan` already exists —
        the caller must :meth:`finalize_plan` or cancel before planning
        the next cycle. Prevents accidental overwrite of in-flight state.
        """
        state = self.load_state()
        if state.active_plan is not None:
            raise RebalanceError(
                "an active_plan already exists; reconcile_and_resume() "
                "or finalize_plan() the prior cycle before planning a "
                "new one"
            )
        asof_iso = _asof_iso(asof)
        plan = plan_rebalance(
            state.current_actual_weights,
            target_weights,
            asof=asof_iso,
            universe_hash=self.universe_hash,
            no_trade_band=self.no_trade_band,
        )
        state.active_plan = plan.to_dict()
        self.write_state(state)
        return plan

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    def execute_plan(
        self,
        plan: RebalancePlan,
        send_order: Callable[[Order], "ExecResult | bool"],
    ) -> List[Order]:
        """Send each non-FILLED order; persist after EVERY attempt.

        ``send_order`` reports an :class:`ExecResult` (or the legacy ``bool``:
        ``True`` == confirmed fill, ``False`` == rejection). The three
        outcomes are kept distinct because conflating broker ACCEPTANCE with
        a confirmed FILL corrupted the actual-weights ledger (C3):

        * :attr:`ExecResult.FILLED` → mark FILLED + book ``fill_weight``.
        * :attr:`ExecResult.ACCEPTED` → mark :attr:`OrderStatus.SENT`
          (in-flight). Fail-closed: NOT counted as a position; a reconcile
          pass promotes SENT→FILLED only when real broker shares confirm.
        * :attr:`ExecResult.REJECTED` → mark FAILED.

        Any exception is logged, persisted as FAILED, and re-raised so the
        caller's retry/halt logic stays in control.

        Persisting after each attempt is the load-bearing invariant for the
        AC: kill mid-loop → restart resumes from the same state → FILLED
        orders never re-send, and a SENT (in-flight) order is NOT resubmitted
        — it awaits fill confirmation, which also avoids a restart
        double-submit.
        """
        touched: List[Order] = []
        for order in plan.orders:
            if order.is_filled:
                continue
            if order.status == OrderStatus.SENT.value:
                # In-flight: broker accepted it, fill not yet confirmed.
                # Resubmitting would risk a double-submit; leave it for the
                # reconcile pass to promote SENT→FILLED (or to re-arm).
                logger.warning(
                    "order %s is SENT (awaiting fill confirmation) — NOT "
                    "resubmitting; reconcile must resolve it",
                    order.client_order_id,
                )
                touched.append(order)
                continue
            try:
                outcome = normalize_exec_result(send_order(order))
            except Exception as exc:
                logger.error(
                    "send_order raised for %s: %s — marking FAILED",
                    order.client_order_id,
                    exc,
                    exc_info=True,
                )
                order.status = OrderStatus.FAILED.value
                self._persist_after_order(plan)
                raise
            if outcome is ExecResult.FILLED:
                order.status = OrderStatus.FILLED.value
                order.fill_weight = float(order.target_weight)
            elif outcome is ExecResult.ACCEPTED:
                order.status = OrderStatus.SENT.value
                logger.info(
                    "order %s ACCEPTED by broker (fill unconfirmed) → SENT; "
                    "ledger unchanged until reconcile confirms the fill",
                    order.client_order_id,
                )
            else:  # REJECTED
                order.status = OrderStatus.FAILED.value
            touched.append(order)
            self._persist_after_order(plan)
        return touched

    def _persist_after_order(self, plan: RebalancePlan) -> None:
        """Snapshot plan + roll up actual weights from the latest fills."""
        state = self.load_state()
        state.active_plan = plan.to_dict()
        actual = dict(state.current_actual_weights)
        for order in plan.orders:
            if not order.is_filled:
                continue
            if order.target_weight == 0.0:
                actual.pop(order.ticker, None)
            else:
                actual[order.ticker] = float(order.target_weight)
        state.current_actual_weights = actual
        self.write_state(state)

    def finalize_plan(self, plan: RebalancePlan) -> None:
        """Mark a fully-filled plan complete; clear ``active_plan``.

        Refuses if any order is still non-FILLED — surface the gap rather
        than silently advancing ``last_rebalance_asof`` past a stuck
        cycle.
        """
        remaining = plan.remaining_orders()
        if remaining:
            raise RebalanceError(
                "cannot finalize plan with "
                f"{len(remaining)} non-filled orders"
            )
        state = self.load_state()
        state.last_rebalance_asof = plan.asof
        state.active_plan = None
        state.current_actual_weights = {
            k: float(v) for k, v in plan.target_weights.items() if v != 0.0
        }
        self.write_state(state)

    # ------------------------------------------------------------------
    # Restart
    # ------------------------------------------------------------------
    def reconcile_and_resume(self) -> Optional[RebalancePlan]:
        """Load the active plan from disk after a restart.

        Returns the :class:`RebalancePlan` with FILLED order statuses
        preserved, or ``None`` if no plan was active when state was
        last persisted. A corrupt plan is logged and cleared (returns
        ``None``) — the caller can then re-plan from the current target.
        """
        state = self.load_state()
        if state.active_plan is None:
            return None
        try:
            return RebalancePlan.from_dict(state.active_plan)
        except (RebalanceError, KeyError, TypeError, ValueError) as exc:
            logger.error(
                "active_plan in %s is corrupt: %s — clearing",
                self.state_path,
                exc,
            )
            state.active_plan = None
            self.write_state(state)
            return None


__all__ = [
    "DEFAULT_NO_TRADE_BAND",
    "DEFAULT_REBALANCE_FREQ_DAYS",
    "Order",
    "OrderStatus",
    "RebalanceError",
    "RebalancePlan",
    "RebalanceScheduler",
    "RebalanceState",
    "SHIP_GATE_PATH_DEFAULT",
    "STATE_PATH_DEFAULT",
    "plan_rebalance",
]
