"""US-012 — IBKR equity order lifecycle (types, TIF, fills, reconciliation, retries).

This module owns the equity-side order envelope: order *type* and
*time-in-force* policy, whole-share rounding with a residual-cash rule,
the SUBMITTED/PARTIAL/FILLED/FAILED state machine, partial-fill
reconciliation, and the retry/backoff envelope around
``broker.place_equity_order(...)``. It is intentionally distinct from the
FX bracket order (``IBKRBroker.place_order``) because:

* Equity rebalances are **agency** market or limit orders against the
  US session, not protective bracket children. There is no SL/TP — the
  rebalance scheduler in :mod:`src.equity.rebalance` owns drift +
  cadence, and the harvester's drawdown circuit-breaker is the
  portfolio-level kill switch (US-008 risk agents). Wrapping equity
  orders in the FX SL/TP/bracket machinery would be wrong: the
  equity book is long-only, the rebalance cadence is the exit, and
  IBKR's child-order bookkeeping for non-bracket equities is much
  thinner.
* Equity orders execute in whole shares — IBKR supports fractional
  shares for some accounts but the harvester intentionally rounds DOWN
  to whole shares so a 1-share rounding error never silently turns a
  PASS gate into a leveraged off-thesis bet. The residual cash from
  rounding stays uninvested and is recycled into the next rebalance.

Order policy (this is the documented contract that US-012 AC #1 calls
out — read this before adding a new order kind):

==========  ============  ===========================================
Order type  TIF default   Use
==========  ============  ===========================================
MARKET      DAY           Periodic rebalance trades; intentionally
                          aggressive — we want the rebalance done in
                          the session it was scheduled for, never
                          rolled past a session boundary. The
                          drawdown circuit-breaker, not GTC, is what
                          gets us out in the tail.
LIMIT       DAY           Same intent as MARKET — DAY-only by default.
                          A LIMIT order that does not fill by the
                          close becomes FAILED at session reconcile;
                          the next rebalance re-evaluates against
                          fresh prices instead of dragging a stale
                          target into a new session.
MARKET      GTC           ONLY for emergency-exit close-everything
                          flow (e.g. the drawdown circuit-breaker
                          firing). GTC is explicit, never default —
                          set ``time_in_force=TIF.GTC`` deliberately.
==========  ============  ===========================================

Whole-share rounding with residual cash rule
--------------------------------------------

For each target weight, the lifecycle computes
``raw_shares = floor(target_weight * NAV / price)`` and surfaces the
residual cash ``leftover = target_weight * NAV - raw_shares * price``.
``leftover`` is NEVER folded into another name. It is reported by
:func:`whole_share_round` and aggregated in
:class:`OrderLifecycle.residual_cash_summary` so the operator (and the
next rebalance plan) can see exactly how much was withheld from the
target book and why.

State machine
-------------

``SUBMITTED -> PARTIAL -> FILLED`` is the happy path. ``FAILED`` is a
terminal sink reachable from any non-FILLED state. Transitions are
validated by :meth:`OrderLifecycle.transition` — illegal transitions
raise :class:`OrderLifecycleError` rather than silently corrupting
state. A forced reconcile mismatch (broker says quantity X, ledger says
quantity Y) drops the order to ``FAILED`` and logs at ERROR level with a
``reason_code`` for the supervisor to grep.

Ship-gate guard
---------------

:func:`enforce_ship_gate` is the same fail-closed pattern used by
:mod:`src.equity.rebalance` and :mod:`src.brokers.ibkr`. The lifecycle
class will refuse to construct unless ``SHIP_GATE.json`` exists, has
``gate_pass==True``, and the persisted ``universe_hash`` matches the
hash of the universe the lifecycle is being attached to.

Retries and BrokerTimeoutError
------------------------------

``place_with_retry`` wraps a single ``broker.place_equity_order(...)``
call with up to :data:`MAX_RETRIES` attempts and exponential backoff
(`backoff = base * 2 ** (attempt - 1)`, clamped). Each attempt is given
to a caller-supplied function returning an
:class:`src.brokers.types.OrderResult`. A transient
:class:`ConnectionError` / :class:`TimeoutError` / :class:`OSError`
triggers retry; any other exception surfaces immediately so
ValueError-class bugs don't get masked behind retry noise. When all
retries are exhausted the lifecycle raises :class:`BrokerTimeoutError`
— a *named* exception so the caller (rebalance scheduler) can match on
``except BrokerTimeoutError`` and decide whether to escalate or wait
for the next session.

Atomic writes and bare-except policy
------------------------------------

Every persisted state mutation uses ``tempfile.mkstemp`` +
``os.replace`` so a crash mid-write leaves the prior valid file on
disk. There are zero bare ``except:`` clauses in this module — every
catch names its exception types and either re-raises or logs and
surfaces. Tests assert this by greping the source.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

from src.brokers.types import OrderResult

logger = logging.getLogger(__name__)


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
STATE_VERSION = 1
MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY_S = 0.05  # tests run fast; production caller can override
DEFAULT_RETRY_MAX_DELAY_S = 30.0


class OrderLifecycleError(RuntimeError):
    """Raised on illegal state transitions or guard refusals.

    Distinct from :class:`BrokerTimeoutError` so retry-vs-config-error
    paths never get tangled at the call site.
    """


class BrokerTimeoutError(RuntimeError):
    """Raised when ``place_with_retry`` exhausts ``MAX_RETRIES``.

    Named (rather than reusing the stdlib :class:`TimeoutError`) so the
    rebalance scheduler can match on ``except BrokerTimeoutError`` and
    distinguish "broker is flapping, retry next session" from "broker
    rejected this specific order, mark it FAILED and move on".
    """

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class OrderState(str, Enum):
    """Equity order lifecycle states.

    The state machine is intentionally small. The rebalance scheduler
    layer above this owns plan-level state (which orders make up which
    rebalance, idempotent restart); this enum just covers the per-order
    journey from "sent to broker" to "broker reports terminal status".
    """

    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    FAILED = "FAILED"


class OrderType(str, Enum):
    """Order-type policy — see module docstring for the matrix."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TIF(str, Enum):
    """Time-in-force policy — see module docstring for the matrix."""

    DAY = "DAY"
    GTC = "GTC"


_LEGAL_TRANSITIONS: Dict[OrderState, frozenset] = {
    OrderState.SUBMITTED: frozenset(
        {OrderState.PARTIAL, OrderState.FILLED, OrderState.FAILED}
    ),
    OrderState.PARTIAL: frozenset({OrderState.FILLED, OrderState.FAILED}),
    # FILLED and FAILED are terminal sinks.
    OrderState.FILLED: frozenset(),
    OrderState.FAILED: frozenset(),
}


@dataclass
class WholeShareDecision:
    """Result of :func:`whole_share_round` for a single target name.

    ``shares`` is the integer quantity the broker should place;
    ``residual_cash`` is the amount the target weight implied that
    could not be expressed in whole shares (always >= 0). The residual
    is reported, never folded into another name.
    """

    ticker: str
    target_weight: float
    nav: float
    price: float
    shares: int
    residual_cash: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EquityOrder:
    """A single equity order tracked through the lifecycle.

    ``client_order_id`` is opaque to this module — it is set by the
    rebalance layer above so restart is idempotent.
    """

    client_order_id: str
    ticker: str
    side: str  # "BUY" or "SELL"
    quantity: int
    order_type: str = OrderType.MARKET.value
    time_in_force: str = TIF.DAY.value
    limit_price: Optional[float] = None
    state: str = OrderState.SUBMITTED.value
    filled_quantity: int = 0
    reason_code: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "EquityOrder":
        return cls(
            client_order_id=str(payload["client_order_id"]),
            ticker=str(payload["ticker"]),
            side=str(payload["side"]),
            quantity=int(payload["quantity"]),  # type: ignore[arg-type]
            order_type=str(payload.get("order_type", OrderType.MARKET.value)),
            time_in_force=str(payload.get("time_in_force", TIF.DAY.value)),
            limit_price=(
                float(payload["limit_price"])  # type: ignore[arg-type]
                if payload.get("limit_price") is not None
                else None
            ),
            state=str(payload.get("state", OrderState.SUBMITTED.value)),
            filled_quantity=int(payload.get("filled_quantity", 0)),  # type: ignore[arg-type]
            reason_code=(
                str(payload["reason_code"])
                if payload.get("reason_code") is not None
                else None
            ),
        )


def _atomic_write_json(payload: dict, path: Path) -> None:
    """Write ``payload`` to ``path`` via ``mkstemp`` + ``os.replace``.

    Mirrors :func:`src.equity.rebalance._atomic_write_json` exactly so
    behavior is identical across the two modules — they share a
    persistence contract.
    """
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


def enforce_ship_gate(path: Path, expected_universe_hash: str) -> None:
    """Refuse unless SHIP_GATE.json clears for THIS universe.

    Fail-closed: missing file, malformed JSON, ``gate_pass`` not
    exactly ``True``, missing/blank/mismatched ``universe_hash`` all
    raise :class:`OrderLifecycleError`. Behavior is identical to
    :func:`src.equity.rebalance._enforce_ship_gate` and
    :func:`src.brokers.ibkr.enforce_equity_ship_gate` so the three
    surfaces reject in the same way.
    """
    path = Path(path)
    if not path.exists():
        raise OrderLifecycleError(
            f"ship gate refuses: file not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrderLifecycleError(
            f"ship gate refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise OrderLifecycleError(
            f"ship gate refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise OrderLifecycleError(
            "ship gate refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to unblock the order lifecycle)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise OrderLifecycleError(
            f"ship gate refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise OrderLifecycleError(
            "ship gate refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"snapshot={expected_universe_hash!r}) at {path}"
        )


def whole_share_round(
    ticker: str,
    target_weight: float,
    nav: float,
    price: float,
) -> WholeShareDecision:
    """Round a target weight to whole shares; surface residual cash.

    Long-only contract: a negative ``target_weight`` is treated as 0
    (the harvester is long-only by hard rule). Zero/negative ``nav``
    or ``price`` raise — silent zero-rounding here would mask upstream
    data corruption.

    Args:
        ticker: Equity ticker (echoed back into the decision payload).
        target_weight: Fraction of NAV to allocate, in [0, 1].
        nav: Account net asset value, > 0.
        price: Latest equity price, > 0.

    Returns:
        A :class:`WholeShareDecision` whose ``residual_cash`` is the
        money the rounding withheld from the target book. The residual
        is per-name and is never folded into another name's quantity.

    Raises:
        OrderLifecycleError: If ``nav`` or ``price`` are not strictly
            positive — these are upstream invariant violations.
    """
    if nav <= 0:
        raise OrderLifecycleError(
            f"whole_share_round refuses: NAV must be > 0 (got {nav!r}) "
            f"for {ticker}"
        )
    if price <= 0:
        raise OrderLifecycleError(
            f"whole_share_round refuses: price must be > 0 (got {price!r}) "
            f"for {ticker}"
        )

    effective_weight = max(0.0, float(target_weight))
    target_cash = effective_weight * float(nav)
    raw_shares = math.floor(target_cash / float(price))
    shares = max(0, int(raw_shares))
    residual = target_cash - shares * float(price)
    # Floating point can drag the residual slightly negative; clamp it
    # so the residual report never looks like a leverage error.
    if residual < 0 and residual > -1e-9:
        residual = 0.0
    return WholeShareDecision(
        ticker=str(ticker),
        target_weight=effective_weight,
        nav=float(nav),
        price=float(price),
        shares=shares,
        residual_cash=float(residual),
    )


class OrderLifecycle:
    """Manage one rebalance cycle's worth of equity orders.

    Construction is fail-closed against the equity ship gate. The
    lifecycle persists its state (orders + per-order status) atomically
    on every mutation so a crash leaves a consistent snapshot a
    restarted process can resume.
    """

    def __init__(
        self,
        universe_hash: str,
        state_path: Path,
        *,
        ship_gate_path: Optional[Path] = None,
        max_retries: int = MAX_RETRIES,
        retry_base_delay_s: float = DEFAULT_RETRY_BASE_DELAY_S,
        retry_max_delay_s: float = DEFAULT_RETRY_MAX_DELAY_S,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(universe_hash, str) or not universe_hash:
            raise OrderLifecycleError(
                "OrderLifecycle requires a non-empty universe_hash"
            )
        gate_path = Path(
            ship_gate_path
            if ship_gate_path is not None
            else SHIP_GATE_PATH_DEFAULT
        )
        enforce_ship_gate(gate_path, universe_hash)
        self.universe_hash = universe_hash
        self.state_path = Path(state_path)
        self.ship_gate_path = gate_path
        self.max_retries = int(max_retries)
        if self.max_retries < 1:
            raise OrderLifecycleError(
                f"max_retries must be >= 1 (got {max_retries!r})"
            )
        self.retry_base_delay_s = float(retry_base_delay_s)
        self.retry_max_delay_s = float(retry_max_delay_s)
        self._sleep_fn = sleep_fn
        self._orders: Dict[str, EquityOrder] = {}
        self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Hydrate from disk if a prior state file exists; else start empty.

        Corrupted state is fail-closed: parsing errors surface as
        :class:`OrderLifecycleError`. Silent recovery here would let
        a half-written ledger silently double-fill on restart.
        """
        if not self.state_path.exists():
            return
        try:
            raw = self.state_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OrderLifecycleError(
                f"cannot read lifecycle state at {self.state_path}: {exc}"
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OrderLifecycleError(
                f"lifecycle state at {self.state_path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise OrderLifecycleError(
                f"lifecycle state at {self.state_path} is not a JSON object"
            )
        stored_hash = payload.get("universe_hash")
        if stored_hash != self.universe_hash:
            raise OrderLifecycleError(
                "lifecycle state universe_hash mismatch: "
                f"on-disk={stored_hash!r}, expected={self.universe_hash!r}"
            )
        orders_payload = payload.get("orders", [])
        if not isinstance(orders_payload, list):
            raise OrderLifecycleError(
                "lifecycle state 'orders' must be a list of order dicts"
            )
        self._orders = {
            str(o["client_order_id"]): EquityOrder.from_dict(o)
            for o in orders_payload
        }

    def _persist(self) -> None:
        """Snapshot the ledger to disk atomically."""
        payload = {
            "version": STATE_VERSION,
            "universe_hash": self.universe_hash,
            "orders": [o.to_dict() for o in self._orders.values()],
        }
        _atomic_write_json(payload, self.state_path)

    # ------------------------------------------------------------------
    # Order registration + state machine
    # ------------------------------------------------------------------

    def register(self, order: EquityOrder) -> EquityOrder:
        """Register a freshly-SUBMITTED order, persist atomically.

        Re-registering a known ``client_order_id`` is the idempotent
        restart path: the on-disk state for that order is preserved,
        not overwritten. This matches the rebalance layer's contract
        that ``client_order_id`` is the dedupe key.
        """
        if order.side not in ("BUY", "SELL"):
            raise OrderLifecycleError(
                f"order side must be BUY or SELL (got {order.side!r})"
            )
        if order.quantity <= 0:
            raise OrderLifecycleError(
                f"order quantity must be > 0 (got {order.quantity!r}) "
                f"for {order.ticker}"
            )
        if order.order_type == OrderType.LIMIT.value and (
            order.limit_price is None or order.limit_price <= 0
        ):
            raise OrderLifecycleError(
                f"LIMIT order requires limit_price > 0 (got "
                f"{order.limit_price!r}) for {order.ticker}"
            )
        if order.time_in_force not in (TIF.DAY.value, TIF.GTC.value):
            raise OrderLifecycleError(
                f"time_in_force must be DAY or GTC (got "
                f"{order.time_in_force!r}) for {order.ticker}"
            )
        if order.client_order_id in self._orders:
            return self._orders[order.client_order_id]
        self._orders[order.client_order_id] = order
        self._persist()
        return order

    def get(self, client_order_id: str) -> EquityOrder:
        if client_order_id not in self._orders:
            raise OrderLifecycleError(
                f"unknown client_order_id {client_order_id!r}"
            )
        return self._orders[client_order_id]

    def all_orders(self) -> List[EquityOrder]:
        return list(self._orders.values())

    def transition(
        self,
        client_order_id: str,
        target: OrderState,
        *,
        reason_code: Optional[str] = None,
        filled_quantity: Optional[int] = None,
    ) -> EquityOrder:
        """Move an order to ``target``; persist atomically.

        Illegal transitions raise :class:`OrderLifecycleError` rather
        than silently corrupting state. A FAILED transition logs at
        ERROR with ``reason_code`` so the supervisor can grep one line
        and find the root cause.
        """
        order = self.get(client_order_id)
        current = OrderState(order.state)
        if target not in _LEGAL_TRANSITIONS[current]:
            raise OrderLifecycleError(
                f"illegal transition for {client_order_id}: "
                f"{current.value} -> {target.value}"
            )
        if target is OrderState.PARTIAL:
            if filled_quantity is None or filled_quantity <= 0:
                raise OrderLifecycleError(
                    "PARTIAL transition requires filled_quantity > 0 "
                    f"(got {filled_quantity!r}) for {client_order_id}"
                )
            if filled_quantity >= order.quantity:
                raise OrderLifecycleError(
                    "PARTIAL transition requires filled_quantity < quantity "
                    f"(got filled={filled_quantity}, "
                    f"quantity={order.quantity}) for {client_order_id}"
                )
            order.filled_quantity = int(filled_quantity)
        elif target is OrderState.FILLED:
            order.filled_quantity = int(
                filled_quantity if filled_quantity is not None
                else order.quantity
            )
            if order.filled_quantity != order.quantity:
                raise OrderLifecycleError(
                    "FILLED transition requires filled_quantity == quantity "
                    f"(got filled={order.filled_quantity}, "
                    f"quantity={order.quantity}) for {client_order_id}"
                )
        elif target is OrderState.FAILED:
            if reason_code is None or not reason_code:
                raise OrderLifecycleError(
                    "FAILED transition requires a non-empty reason_code"
                )
            order.reason_code = reason_code
            logger.error(
                "equity order FAILED: client_order_id=%s ticker=%s "
                "side=%s qty=%d reason_code=%s",
                client_order_id,
                order.ticker,
                order.side,
                order.quantity,
                reason_code,
            )
            if filled_quantity is not None:
                order.filled_quantity = int(filled_quantity)
        order.state = target.value
        self._persist()
        return order

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(
        self,
        client_order_id: str,
        broker_filled_quantity: int,
    ) -> EquityOrder:
        """Reconcile a tracked order against the broker's reported fills.

        Behavior:
        * ``broker_filled_quantity == order.quantity`` -> FILLED
        * ``0 < broker_filled_quantity < order.quantity`` -> PARTIAL
        * ``broker_filled_quantity == 0 and state == SUBMITTED`` -> no
          transition (still SUBMITTED, awaiting fill)
        * ``broker_filled_quantity < order.filled_quantity`` or
          ``broker_filled_quantity > order.quantity`` -> mismatch ->
          FAILED with ``reason_code=reconcile_mismatch``; logged at ERROR.
        * Reconciling a terminal FILLED order with the same quantity is
          a no-op; any other terminal-state reconcile is a mismatch.
        """
        order = self.get(client_order_id)
        broker_filled_quantity = int(broker_filled_quantity)
        if broker_filled_quantity < 0:
            self.transition(
                client_order_id,
                OrderState.FAILED,
                reason_code="reconcile_negative_qty",
                filled_quantity=order.filled_quantity,
            )
            return self.get(client_order_id)

        current = OrderState(order.state)

        if current is OrderState.FILLED:
            if broker_filled_quantity == order.quantity:
                return order  # consistent no-op
            self._force_fail(
                client_order_id,
                reason="reconcile_mismatch",
                broker_qty=broker_filled_quantity,
                ledger_qty=order.filled_quantity,
            )
            return self.get(client_order_id)

        if current is OrderState.FAILED:
            # Terminal already; any reconcile attempt other than an
            # exact match on filled_quantity is logged but does not
            # re-transition (FAILED is terminal).
            return order

        # Currently SUBMITTED or PARTIAL.
        if broker_filled_quantity > order.quantity:
            self._force_fail(
                client_order_id,
                reason="reconcile_mismatch",
                broker_qty=broker_filled_quantity,
                ledger_qty=order.filled_quantity,
            )
            return self.get(client_order_id)
        if broker_filled_quantity < order.filled_quantity:
            # Ledger says we already filled more than the broker now
            # claims — that's a hard inconsistency, never recover silently.
            self._force_fail(
                client_order_id,
                reason="reconcile_mismatch",
                broker_qty=broker_filled_quantity,
                ledger_qty=order.filled_quantity,
            )
            return self.get(client_order_id)
        if broker_filled_quantity == 0:
            return order  # no fill yet
        if broker_filled_quantity == order.quantity:
            target = OrderState.FILLED
        else:
            target = OrderState.PARTIAL
        return self.transition(
            client_order_id,
            target,
            filled_quantity=broker_filled_quantity,
        )

    def _force_fail(
        self,
        client_order_id: str,
        *,
        reason: str,
        broker_qty: int,
        ledger_qty: int,
    ) -> None:
        """Drop an order to FAILED out of band of normal transitions.

        Used by :meth:`reconcile` when the broker's reported quantity
        contradicts the ledger. We bypass :meth:`transition`'s
        legal-transition guard because FILLED -> FAILED is intentionally
        forbidden in the happy path but is mandatory for a mismatch:
        the alternative is leaving a known-bad order in a "successful"
        terminal state.
        """
        order = self.get(client_order_id)
        order.state = OrderState.FAILED.value
        order.reason_code = reason
        self._persist()
        logger.error(
            "equity order reconcile mismatch: client_order_id=%s ticker=%s "
            "broker_filled=%d ledger_filled=%d ordered_qty=%d "
            "reason_code=%s",
            client_order_id,
            order.ticker,
            broker_qty,
            ledger_qty,
            order.quantity,
            reason,
        )

    def residual_cash_summary(
        self,
        decisions: List[WholeShareDecision],
    ) -> Dict[str, float]:
        """Aggregate per-name residual cash from rounding.

        Returns a tiny rollup so the rebalance layer can persist it
        alongside the rebalance plan: ``{"total": float,
        "per_ticker": {...}}``. Residuals are NOT folded into another
        name; this is purely a report.
        """
        per_ticker = {d.ticker: float(d.residual_cash) for d in decisions}
        total = float(sum(per_ticker.values()))
        return {"total": total, "per_ticker": per_ticker}

    # ------------------------------------------------------------------
    # Retry envelope
    # ------------------------------------------------------------------

    def place_with_retry(
        self,
        send: Callable[[], OrderResult],
    ) -> OrderResult:
        """Call ``send()`` with retry + exponential backoff.

        Retries on :class:`ConnectionError`, :class:`TimeoutError`,
        :class:`OSError`. Any other exception surfaces immediately so
        bugs aren't masked by the retry envelope. After
        :attr:`max_retries` attempts, raises :class:`BrokerTimeoutError`
        carrying the attempt count.

        Note this does NOT touch ledger state — the caller decides
        what to do with the returned :class:`OrderResult` (e.g. transition
        to SUBMITTED, FAILED with reason_code, etc.). Keeping the retry
        envelope free of side effects means it composes cleanly with the
        state machine and is trivial to test.
        """
        attempt = 0
        last_exc: Optional[BaseException] = None
        while attempt < self.max_retries:
            attempt += 1
            try:
                return send()
            except (ConnectionError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_delay_s * (2 ** (attempt - 1))
                delay = min(delay, self.retry_max_delay_s)
                jitter = random.uniform(0, delay * 0.1)
                delay = delay + jitter
                logger.warning(
                    "place_with_retry: attempt %d/%d failed (%s: %s); "
                    "retrying in %.3fs",
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                self._sleep_fn(delay)
        msg = (
            f"broker exhausted {self.max_retries} retries; "
            f"last error: {type(last_exc).__name__}: {last_exc}"
        )
        logger.error(msg)
        raise BrokerTimeoutError(msg, attempts=attempt) from last_exc


# Re-export the canonical asof-stamped state-file path helper. The
# rebalance layer chooses the per-cycle file; the lifecycle accepts any
# path the operator points it at.
__all__ = [
    "BrokerTimeoutError",
    "EquityOrder",
    "MAX_RETRIES",
    "OrderLifecycle",
    "OrderLifecycleError",
    "OrderState",
    "OrderType",
    "SHIP_GATE_PATH_DEFAULT",
    "STATE_VERSION",
    "TIF",
    "WholeShareDecision",
    "enforce_ship_gate",
    "whole_share_round",
]
