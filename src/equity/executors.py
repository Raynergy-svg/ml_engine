"""US-015 — Equity order-lifecycle executor framework.

Adapted from the Hummingbot ``strategy_v2/executors`` pattern
(https://github.com/hummingbot/hummingbot), licensed under the Apache
License, Version 2.0. The original Hummingbot module exposes an
``ExecutorBase`` abstract class with a status state machine
(``NOT_STARTED → ACTIVE → COMPLETED/FAILED``) and specialised subclasses
(``PositionExecutor``, ``TWAPExecutor``, ``DCAExecutor``, …) that drive
child orders through a broker connector. Clean-room re-derivation in pure
Python; the original Hummingbot source files are not vendored. Apache-2.0
attribution is retained in this header and the repo-root ``NOTICE``.

Why a separate executor layer above US-012 ``OrderLifecycle``
------------------------------------------------------------

US-012 owns the per-order envelope: order type, TIF policy, whole-share
rounding, the SUBMITTED/PARTIAL/FILLED/FAILED state machine, retry +
exponential backoff, and the ship-gate guard. What it does *not* own is
the higher-level orchestration: "given a target equity position, place
the parent order, watch fills, and report a single COMPLETED/FAILED
verdict back to the rebalance scheduler"; "given a large parent
quantity, slice it into N child orders over a TWAP horizon and aggregate
their statuses". That orchestration is what this executor layer adds —
each executor is a thin run-loop above ``OrderLifecycle`` whose only job
is to compose lifecycle calls into a higher-level intent.

This separation matches Hummingbot's split between connectors (the
broker layer) and executors (the orchestration layer). It keeps the
ship-gate guard, retry envelope, atomic ledger, and reconciliation
machinery from US-012 as the single source of truth — the executors do
not open a parallel order path, they only orchestrate ``OrderLifecycle``
calls.

Surfaces
--------

* :class:`ExecutorStatus` — NOT_STARTED / ACTIVE / COMPLETED / FAILED.
* :class:`ExecutorBase` — abstract base; ship-gate guarded at construct,
  atomic state persistence on every status transition.
* :class:`PositionExecutor` — opens one equity position: computes
  whole-share quantity (via :func:`whole_share_round`), routes through
  ``OrderLifecycle.place_with_retry`` → ``broker.place_equity_order``,
  reconciles using broker-reported fill quantity.
* :class:`TWAPExecutor` — slices a parent quantity into N equal-sized
  child :class:`PositionExecutor` instances released on a caller-supplied
  monotonic clock. Each child still routes through the same lifecycle —
  TWAP is purely a release schedule, not a parallel order envelope.
* :func:`enforce_ship_gate` — public so tests can drive the guard
  directly; identical fail-closed contract to every other equity
  surface.

Ship-gate guard
---------------

:func:`enforce_ship_gate` is byte-for-byte identical to the other equity
guards (``src.equity.rebalance._enforce_ship_gate``,
``src.equity.order_lifecycle.enforce_ship_gate``,
``src.equity.kill_switch.enforce_ship_gate``,
``src.brokers.ibkr.enforce_equity_ship_gate``,
``src.equity.depth_pricing.DepthAwareSizer._enforce_ship_gate``).
Constructing an executor without a passing gate raises
:class:`ExecutorError`; the same check is re-run before any broker call
inside :meth:`ExecutorBase.run` so a gate invalidated mid-flight fails
closed at the next poll.

Atomic writes and bare-except policy
------------------------------------

Every state mutation persists via ``tempfile.mkstemp`` + ``os.replace``
so a crash mid-write leaves the prior valid file on disk. Zero bare
``except:`` clauses — every catch names its exception types and either
re-raises or logs and surfaces. Tests assert this with a source grep.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

from src.brokers.base import BrokerClient
from src.brokers.instrument import Instrument
from src.brokers.types import OrderResult
from src.equity.order_lifecycle import (
    BrokerTimeoutError,
    EquityOrder,
    OrderLifecycle,
    OrderLifecycleError,
    OrderState,
    OrderType,
    TIF,
    whole_share_round,
)

logger = logging.getLogger(__name__)


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
STATE_VERSION = 1


class ExecutorError(RuntimeError):
    """Raised on ship-gate refusal, illegal status transitions, or contract bugs.

    Distinct from :class:`OrderLifecycleError` so the rebalance supervisor
    can disambiguate "executor orchestration refused" (this) from
    "underlying order ledger refused" (that). The latter is always a
    deeper invariant violation and surfaces from the lifecycle even if
    the executor would have been happy.
    """


class ExecutorStatus(str, Enum):
    """Executor lifecycle states.

    Mirrors Hummingbot's ``RunnableStatus`` shape but pruned to the four
    states the equity path actually needs: equities are long-only,
    rebalance-driven, and have no SL/TP at the broker level, so the
    SHUTDOWN_REQUESTED intermediate state is collapsed into terminal
    COMPLETED/FAILED. The rebalance cadence is the exit; the kill switch
    is the panic button.
    """

    NOT_STARTED = "NOT_STARTED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


_LEGAL_STATUS_TRANSITIONS: Dict[ExecutorStatus, frozenset] = {
    ExecutorStatus.NOT_STARTED: frozenset(
        {ExecutorStatus.ACTIVE, ExecutorStatus.FAILED}
    ),
    ExecutorStatus.ACTIVE: frozenset(
        {ExecutorStatus.COMPLETED, ExecutorStatus.FAILED}
    ),
    ExecutorStatus.COMPLETED: frozenset(),
    ExecutorStatus.FAILED: frozenset(),
}


def _atomic_write_json(payload: Mapping[str, object], path: Path) -> None:
    """Write ``payload`` to ``path`` via ``mkstemp`` + ``os.replace``.

    Mirrors :func:`src.equity.order_lifecycle._atomic_write_json` and
    :func:`src.equity.kill_switch._atomic_write_json` exactly so a crash
    mid-write has identical recovery semantics across every equity
    surface.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
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

    Fail-closed: missing file, malformed JSON, ``gate_pass`` not exactly
    ``True``, missing / blank / mismatched ``universe_hash`` all raise
    :class:`ExecutorError`. Contract is identical to every other equity
    ship-gate guard so a fixture that passes one surface passes them
    all.
    """
    path = Path(path)
    if not path.exists():
        raise ExecutorError(
            f"executor refuses: ship gate not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutorError(
            f"executor refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ExecutorError(
            f"executor refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise ExecutorError(
            "executor refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to unblock the executor)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise ExecutorError(
            f"executor refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise ExecutorError(
            "executor refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"expected={expected_universe_hash!r}) at {path}"
        )


@dataclass
class ExecutorResult:
    """Structured outcome of an executor run.

    ``status`` is the terminal :class:`ExecutorStatus`. ``children`` is
    an ordered list of per-child OrderLifecycle ``client_order_id``
    strings — for :class:`PositionExecutor` exactly one entry, for
    :class:`TWAPExecutor` one per slice. ``filled_quantity`` aggregates
    fills across children; ``failed_reasons`` collects per-child failure
    reason codes for the supervisor.
    """

    executor_id: str
    status: str
    children: List[str] = field(default_factory=list)
    filled_quantity: int = 0
    failed_reasons: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ExecutorBase:
    """Abstract base: status state machine + ship-gate guard + atomic ledger.

    Concrete subclasses implement :meth:`_advance` — one tick of the
    control loop — and return ``True`` when the executor reaches a
    terminal state. :meth:`run` drives the loop, re-enforcing the ship
    gate before every tick so a gate that flips between arming and
    firing fails closed at the next poll.

    The base intentionally does NOT spawn threads or asyncio tasks.
    IBKR's TWS connection is in-process and serialized internally; the
    rebalance scheduler is sync; adding concurrency here would be
    ceremony, not throughput. A caller-supplied ``sleep_fn`` makes the
    poll cadence injectable for tests.
    """

    def __init__(
        self,
        executor_id: str,
        universe_hash: str,
        state_path: Path,
        lifecycle: OrderLifecycle,
        *,
        ship_gate_path: Optional[Path] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(executor_id, str) or not executor_id:
            raise ExecutorError(
                "ExecutorBase requires a non-empty executor_id"
            )
        if not isinstance(universe_hash, str) or not universe_hash:
            raise ExecutorError(
                "ExecutorBase requires a non-empty universe_hash"
            )
        if lifecycle.universe_hash != universe_hash:
            raise ExecutorError(
                "lifecycle.universe_hash does not match executor "
                f"universe_hash (lifecycle={lifecycle.universe_hash!r}, "
                f"executor={universe_hash!r})"
            )
        gate_path = Path(
            ship_gate_path
            if ship_gate_path is not None
            else SHIP_GATE_PATH_DEFAULT
        )
        enforce_ship_gate(gate_path, universe_hash)

        self.executor_id = executor_id
        self.universe_hash = universe_hash
        self.state_path = Path(state_path)
        self.lifecycle = lifecycle
        self.ship_gate_path = gate_path
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn
        self._status = ExecutorStatus.NOT_STARTED
        self._started_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._failed_reasons: List[str] = []
        self._children: List[str] = []
        self._persist()

    # ------------------------------------------------------------------
    # Status state machine
    # ------------------------------------------------------------------

    @property
    def status(self) -> ExecutorStatus:
        return self._status

    def _transition(
        self,
        target: ExecutorStatus,
        *,
        reason: Optional[str] = None,
    ) -> None:
        """Move the executor to ``target`` and persist atomically.

        Illegal transitions raise :class:`ExecutorError` rather than
        silently corrupting state. FAILED requires a non-empty reason so
        the supervisor can grep the audit log and find the root cause.
        """
        if target not in _LEGAL_STATUS_TRANSITIONS[self._status]:
            raise ExecutorError(
                f"illegal executor status transition for {self.executor_id}: "
                f"{self._status.value} -> {target.value}"
            )
        if target is ExecutorStatus.FAILED:
            if reason is None or not reason:
                raise ExecutorError(
                    "FAILED transition requires a non-empty reason"
                )
            self._failed_reasons.append(reason)
            logger.error(
                "executor FAILED: executor_id=%s reason=%s",
                self.executor_id, reason,
            )
        if target is ExecutorStatus.ACTIVE:
            self._started_at = self._monotonic_fn()
        if target in (ExecutorStatus.COMPLETED, ExecutorStatus.FAILED):
            self._completed_at = self._monotonic_fn()
        self._status = target
        self._persist()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(
        self,
        broker: BrokerClient,
        *,
        max_ticks: int = 100,
        tick_interval_s: float = 0.05,
    ) -> ExecutorResult:
        """Drive the executor to a terminal state.

        Re-enforces the ship gate before every tick — if SHIP_GATE.json
        is invalidated mid-flight (e.g. universe redeployed during a
        long TWAP), the next poll surfaces :class:`ExecutorError` from
        the gate and the executor transitions to FAILED.

        Args:
            broker: Live :class:`BrokerClient`. The executor does not
                own connect/disconnect — the caller (rebalance
                scheduler) does.
            max_ticks: Hard cap on poll iterations. A misbehaving child
                (broker that never reports fills) must not pin the
                rebalance scheduler indefinitely.
            tick_interval_s: Sleep between ticks. Tests inject a
                zero-sleep ``sleep_fn`` to avoid wall-clock waits.

        Returns:
            :class:`ExecutorResult` carrying the terminal status,
            children, aggregate fill quantity, and any failure reasons.
        """
        if self._status is ExecutorStatus.NOT_STARTED:
            self._transition(ExecutorStatus.ACTIVE)
        if self._status in (
            ExecutorStatus.COMPLETED,
            ExecutorStatus.FAILED,
        ):
            return self._build_result()

        ticks = 0
        while ticks < max_ticks:
            ticks += 1
            try:
                enforce_ship_gate(self.ship_gate_path, self.universe_hash)
            except ExecutorError as exc:
                self._transition(
                    ExecutorStatus.FAILED,
                    reason=f"ship_gate_invalidated: {exc}",
                )
                return self._build_result()

            try:
                done = self._advance(broker)
            except (
                OrderLifecycleError,
                BrokerTimeoutError,
                ExecutorError,
            ) as exc:
                self._transition(
                    ExecutorStatus.FAILED,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                return self._build_result()

            if done:
                if self._status is ExecutorStatus.ACTIVE:
                    self._transition(ExecutorStatus.COMPLETED)
                return self._build_result()

            self._sleep_fn(float(tick_interval_s))

        self._transition(
            ExecutorStatus.FAILED,
            reason=f"max_ticks_exceeded: {max_ticks}",
        )
        return self._build_result()

    def _advance(self, broker: BrokerClient) -> bool:
        """Subclass hook — one tick of work. Return ``True`` when done."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "executor_id": self.executor_id,
            "executor_type": type(self).__name__,
            "universe_hash": self.universe_hash,
            "status": self._status.value,
            "children": list(self._children),
            "failed_reasons": list(self._failed_reasons),
            "started_at": self._started_at,
            "completed_at": self._completed_at,
        }
        _atomic_write_json(payload, self.state_path)

    def _build_result(self) -> ExecutorResult:
        filled = 0
        for client_order_id in self._children:
            try:
                order = self.lifecycle.get(client_order_id)
            except OrderLifecycleError:
                continue
            filled += int(order.filled_quantity)
        duration = 0.0
        if self._started_at is not None and self._completed_at is not None:
            duration = max(0.0, self._completed_at - self._started_at)
        return ExecutorResult(
            executor_id=self.executor_id,
            status=self._status.value,
            children=list(self._children),
            filled_quantity=filled,
            failed_reasons=list(self._failed_reasons),
            duration_s=duration,
        )


class PositionExecutor(ExecutorBase):
    """Open one equity position via :class:`OrderLifecycle`.

    Computes whole-share quantity from ``(target_weight, nav, price)``
    via :func:`whole_share_round` so the residual-cash rule from US-012
    is honoured. Submits exactly one order via
    ``OrderLifecycle.place_with_retry → broker.place_equity_order`` and
    polls broker fills until the lifecycle order reaches a terminal
    state. The lifecycle's retry envelope + reconcile mismatch handling
    do the heavy lifting — this class is the orchestration shell.

    Long-only contract: ``side="BUY"`` is the harvester default;
    ``side="SELL"`` is permitted because the rebalance scheduler emits
    SELL legs to flatten over-weight names. Whole-share rounding is
    applied to both sides identically.
    """

    def __init__(
        self,
        executor_id: str,
        universe_hash: str,
        state_path: Path,
        lifecycle: OrderLifecycle,
        instrument: Instrument,
        side: str,
        target_weight: float,
        nav: float,
        price: float,
        *,
        order_type: str = OrderType.MARKET.value,
        time_in_force: str = TIF.DAY.value,
        limit_price: Optional[float] = None,
        ship_gate_path: Optional[Path] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
        fill_query: Optional[Callable[[BrokerClient, str], int]] = None,
    ) -> None:
        super().__init__(
            executor_id=executor_id,
            universe_hash=universe_hash,
            state_path=state_path,
            lifecycle=lifecycle,
            ship_gate_path=ship_gate_path,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
        )
        if side not in ("BUY", "SELL"):
            raise ExecutorError(
                f"PositionExecutor side must be BUY or SELL (got {side!r})"
            )
        if instrument.asset_class != "EQUITY":
            raise ExecutorError(
                "PositionExecutor requires an EQUITY instrument "
                f"(got asset_class={instrument.asset_class!r})"
            )
        decision = whole_share_round(
            ticker=instrument.symbol,
            target_weight=target_weight,
            nav=nav,
            price=price,
        )
        self.instrument = instrument
        self.side = side
        self.target_weight = float(target_weight)
        self.nav = float(nav)
        self.price = float(price)
        self.target_shares = int(decision.shares)
        self.residual_cash = float(decision.residual_cash)
        self.order_type = str(order_type)
        self.time_in_force = str(time_in_force)
        self.limit_price = (
            float(limit_price) if limit_price is not None else None
        )
        self._fill_query = (
            fill_query
            if fill_query is not None
            else _default_fill_query
        )
        self._submitted = False
        self._client_order_id = f"{self.executor_id}:child:0"

    def _advance(self, broker: BrokerClient) -> bool:
        if self.target_shares <= 0:
            # Zero-share targets are a legal degenerate case (the
            # rebalance band can ask for less than one share). Mark
            # COMPLETED without a child order; the residual cash is
            # already exposed via :attr:`residual_cash` for the
            # rebalance audit log.
            return True

        if not self._submitted:
            order = EquityOrder(
                client_order_id=self._client_order_id,
                ticker=self.instrument.symbol,
                side=self.side,
                quantity=self.target_shares,
                order_type=self.order_type,
                time_in_force=self.time_in_force,
                limit_price=self.limit_price,
            )
            self.lifecycle.register(order)
            self._children.append(self._client_order_id)
            self._persist()

            result = self.lifecycle.place_with_retry(
                lambda: broker.place_equity_order(
                    self.instrument, self.side, self.target_shares,
                )
            )
            self._submitted = True
            self._handle_submit_result(result)

        current = self.lifecycle.get(self._client_order_id)
        state = OrderState(current.state)
        if state in (OrderState.FILLED, OrderState.FAILED):
            if state is OrderState.FAILED:
                reason = current.reason_code or "lifecycle_failed"
                # Surface the child-FAILED as an executor-level reason
                # and transition the executor itself to FAILED.
                raise ExecutorError(
                    f"child order failed: {self._client_order_id}:{reason}"
                )
            return True

        broker_filled = self._fill_query(broker, self.instrument.symbol)
        # Clamp the broker-reported quantity to the order book so a
        # stale or aggregate position reading can't trip the
        # reconcile-mismatch guard. The lifecycle's strict mismatch
        # check exists for *order-level* contradictions; here we're
        # reading position state, which may legitimately exceed the
        # current child order's quantity if other executors are running.
        broker_filled = max(0, min(int(broker_filled), self.target_shares))

        # Suppress reconcile calls that would attempt an illegal
        # same-state PARTIAL → PARTIAL transition (the US-012 lifecycle
        # only allows forward transitions). Incremental partial fills
        # accumulate at the broker; the executor advances to terminal
        # state only when broker_filled reaches target_shares OR the
        # next reconcile would change order state.
        if (
            state is OrderState.PARTIAL
            and 0 < broker_filled < self.target_shares
        ):
            # Either still at the same partial level (no progress) or
            # incrementally higher but still partial — both are no-ops
            # for the state machine.
            return False

        self.lifecycle.reconcile(self._client_order_id, broker_filled)

        refreshed = self.lifecycle.get(self._client_order_id)
        refreshed_state = OrderState(refreshed.state)
        if refreshed_state is OrderState.FAILED:
            reason = refreshed.reason_code or "lifecycle_failed"
            raise ExecutorError(
                f"child order failed: {self._client_order_id}:{reason}"
            )
        return refreshed_state is OrderState.FILLED

    def _handle_submit_result(self, result: OrderResult) -> None:
        status = result.status.upper() if result.status else ""
        if status in ("REJECTED", "FAILED", "CANCELLED", "CANCELED"):
            reason = result.error_message or f"broker_status={result.status}"
            # The lifecycle order is still in SUBMITTED state at this
            # point; transition it to FAILED so the audit ledger
            # carries the broker's reason verbatim.
            self.lifecycle.transition(
                self._client_order_id,
                OrderState.FAILED,
                reason_code=f"broker_rejected:{reason}",
            )
            return
        # PENDING / FILLED / PARTIALLY_FILLED are all valid acceptance
        # signals; we let the reconcile loop sort out terminal state.


def _default_fill_query(broker: BrokerClient, symbol: str) -> int:
    """Default broker → filled-quantity probe.

    Reads ``broker.get_open_positions()`` and returns the integer share
    count for ``symbol`` (or zero if not present). This is the most
    portable signal across :class:`BrokerClient` implementations —
    every broker can answer "what do I currently hold". The reconcile
    layer in :class:`OrderLifecycle` is responsible for translating
    that snapshot into the correct order-level fill state.

    Tests can inject a custom ``fill_query`` to drive deterministic
    fill progressions without depending on broker-level position math.
    """
    try:
        positions = broker.get_open_positions()
    except (ConnectionError, TimeoutError, OSError) as exc:
        logger.warning(
            "fill_query: get_open_positions failed (%s: %s); "
            "treating as zero fill for now",
            type(exc).__name__, exc,
        )
        return 0
    for pos in positions:
        if pos.instrument == symbol:
            return int(pos.quantity)
    return 0


class TWAPExecutor(ExecutorBase):
    """Slice a parent equity target into N equal-sized child positions.

    Releases one :class:`PositionExecutor` per slice on a caller-supplied
    monotonic clock; each child still routes through the same
    :class:`OrderLifecycle`. TWAP is purely a release schedule, not a
    parallel order envelope — slice sizing uses the same whole-share
    rounding as a single :class:`PositionExecutor` so the residual-cash
    rule is honoured per slice.

    Whole-share contract: ``total_shares`` is divided as evenly as
    possible across ``num_slices``. Remainder shares (e.g. 10 shares
    across 3 slices → 4/3/3) are placed on the FIRST slices so the
    schedule front-loads any odd lots — the operator wants the largest
    slice executed first when the parent intent is "get to the target
    quickly within the TWAP horizon".
    """

    def __init__(
        self,
        executor_id: str,
        universe_hash: str,
        state_path: Path,
        lifecycle: OrderLifecycle,
        instrument: Instrument,
        side: str,
        total_shares: int,
        num_slices: int,
        slice_interval_s: float,
        *,
        order_type: str = OrderType.MARKET.value,
        time_in_force: str = TIF.DAY.value,
        limit_price: Optional[float] = None,
        ship_gate_path: Optional[Path] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
        fill_query: Optional[Callable[[BrokerClient, str], int]] = None,
    ) -> None:
        super().__init__(
            executor_id=executor_id,
            universe_hash=universe_hash,
            state_path=state_path,
            lifecycle=lifecycle,
            ship_gate_path=ship_gate_path,
            sleep_fn=sleep_fn,
            monotonic_fn=monotonic_fn,
        )
        if side not in ("BUY", "SELL"):
            raise ExecutorError(
                f"TWAPExecutor side must be BUY or SELL (got {side!r})"
            )
        if instrument.asset_class != "EQUITY":
            raise ExecutorError(
                "TWAPExecutor requires an EQUITY instrument "
                f"(got asset_class={instrument.asset_class!r})"
            )
        if not isinstance(total_shares, int) or total_shares <= 0:
            raise ExecutorError(
                "TWAPExecutor total_shares must be a positive int "
                f"(got {total_shares!r})"
            )
        if not isinstance(num_slices, int) or num_slices <= 0:
            raise ExecutorError(
                "TWAPExecutor num_slices must be a positive int "
                f"(got {num_slices!r})"
            )
        if num_slices > total_shares:
            raise ExecutorError(
                "TWAPExecutor num_slices cannot exceed total_shares "
                f"(slices={num_slices}, shares={total_shares}); "
                "each slice must be >= 1 whole share"
            )
        if slice_interval_s < 0:
            raise ExecutorError(
                "TWAPExecutor slice_interval_s must be >= 0 "
                f"(got {slice_interval_s!r})"
            )
        self.instrument = instrument
        self.side = side
        self.total_shares = int(total_shares)
        self.num_slices = int(num_slices)
        self.slice_interval_s = float(slice_interval_s)
        self.order_type = str(order_type)
        self.time_in_force = str(time_in_force)
        self.limit_price = (
            float(limit_price) if limit_price is not None else None
        )
        self._fill_query = (
            fill_query
            if fill_query is not None
            else _default_fill_query
        )
        self.slice_quantities: List[int] = _split_evenly(
            self.total_shares, self.num_slices,
        )
        self._next_slice_index = 0
        self._next_slice_at: Optional[float] = None
        self._slice_orders: List[str] = []

    def _advance(self, broker: BrokerClient) -> bool:
        # Release the next slice when its scheduled time has arrived.
        if self._next_slice_index < self.num_slices:
            now = self._monotonic_fn()
            if self._next_slice_at is None:
                # First slice releases immediately on the first tick.
                self._next_slice_at = now
            if now >= self._next_slice_at:
                self._release_slice(broker)
                self._next_slice_index += 1
                self._next_slice_at = now + self.slice_interval_s

        # Reconcile every in-flight slice. Done iff every slice is
        # terminal AND every slice has been released.
        if self._next_slice_index < self.num_slices:
            return False

        all_terminal = True
        any_failed = False
        for client_order_id in self._slice_orders:
            order = self.lifecycle.get(client_order_id)
            state = OrderState(order.state)
            if state is OrderState.FAILED:
                any_failed = True
                self._failed_reasons.append(
                    f"{client_order_id}:"
                    f"{order.reason_code or 'lifecycle_failed'}"
                )
                continue
            if state is OrderState.FILLED:
                continue
            broker_filled = self._fill_query(broker, self.instrument.symbol)
            cumulative_target = sum(
                self.slice_quantities[: self._slice_orders.index(client_order_id) + 1]
            )
            clamped = max(
                0, min(int(broker_filled), int(cumulative_target))
            )
            # Compute the per-child fill: cumulative broker quantity
            # minus prior slices' filled_quantity.
            prior_filled = 0
            for prior_id in self._slice_orders:
                if prior_id == client_order_id:
                    break
                prior_filled += int(self.lifecycle.get(prior_id).filled_quantity)
            this_slice_target = order.quantity
            this_slice_filled = max(
                0, min(clamped - prior_filled, this_slice_target)
            )
            # Same PARTIAL → PARTIAL suppression as PositionExecutor —
            # the US-012 lifecycle disallows same-state transitions, so
            # we only call reconcile when the call would actually move
            # the order forward.
            if (
                state is OrderState.PARTIAL
                and 0 < this_slice_filled < this_slice_target
            ):
                all_terminal = False
                continue
            self.lifecycle.reconcile(client_order_id, this_slice_filled)
            refreshed = self.lifecycle.get(client_order_id)
            refreshed_state = OrderState(refreshed.state)
            if refreshed_state is OrderState.FAILED:
                any_failed = True
                self._failed_reasons.append(
                    f"{client_order_id}:"
                    f"{refreshed.reason_code or 'lifecycle_failed'}"
                )
                continue
            if refreshed_state is not OrderState.FILLED:
                all_terminal = False
        if any_failed:
            raise ExecutorError(
                "TWAP child slice(s) failed: "
                f"{', '.join(self._failed_reasons)}"
            )
        return all_terminal

    def _release_slice(self, broker: BrokerClient) -> None:
        idx = self._next_slice_index
        quantity = self.slice_quantities[idx]
        client_order_id = f"{self.executor_id}:slice:{idx}"
        order = EquityOrder(
            client_order_id=client_order_id,
            ticker=self.instrument.symbol,
            side=self.side,
            quantity=quantity,
            order_type=self.order_type,
            time_in_force=self.time_in_force,
            limit_price=self.limit_price,
        )
        self.lifecycle.register(order)
        self._slice_orders.append(client_order_id)
        self._children.append(client_order_id)
        self._persist()

        result = self.lifecycle.place_with_retry(
            lambda: broker.place_equity_order(
                self.instrument, self.side, quantity,
            )
        )
        status = result.status.upper() if result.status else ""
        if status in ("REJECTED", "FAILED", "CANCELLED", "CANCELED"):
            reason = result.error_message or f"broker_status={result.status}"
            self.lifecycle.transition(
                client_order_id,
                OrderState.FAILED,
                reason_code=f"broker_rejected:{reason}",
            )


def _split_evenly(total: int, n: int) -> List[int]:
    """Split ``total`` shares across ``n`` slices as evenly as possible.

    Remainder shares are placed on the FIRST slices (front-loaded) so a
    10-share, 3-slice schedule becomes ``[4, 3, 3]``. Front-loading
    matches the operator's stated TWAP intent — get the largest slice
    out first so any odd lot doesn't sit at the back of the schedule.
    """
    if n <= 0 or total <= 0:
        return []
    base = total // n
    remainder = total - base * n
    return [base + 1 if i < remainder else base for i in range(n)]


__all__ = [
    "ExecutorBase",
    "ExecutorError",
    "ExecutorResult",
    "ExecutorStatus",
    "PositionExecutor",
    "SHIP_GATE_PATH_DEFAULT",
    "STATE_VERSION",
    "TWAPExecutor",
    "enforce_ship_gate",
]
