"""US-016 — Shadow fill simulator.

Models broker fills for a US-012 :class:`~src.equity.order_lifecycle.EquityOrder`
against a US-014 :class:`~src.equity.depth_pricing.L2OrderBook` (or a plain
top-of-book price + spread when an L2 snapshot is not available), so the
US-017 autonomous control loop and the US-021 shadow / paper run can be
driven end-to-end without ever sending a live order. Every input is
deterministic — given the same order, the same book, and the same
:class:`SlippageModel`, the simulator emits the same
:class:`SimulatedFill` bytewise. That property is the whole point of
"shadow mode": replays of yesterday's tape MUST land on the same fill
prices today, otherwise the loop smoke test can't diff what changed
between runs.

Surfaces
--------

* :class:`SlippageModel` — frozen dataclass holding the three knobs we
  expose: ``extra_slippage_bps`` (multiplicative haircut on the VWAP to
  model queue-position / latency that the visible L2 cannot see),
  ``participation_rate`` (cap fill quantity at a fraction of visible
  depth, used by US-017 to refuse to over-trade thin books), and
  ``fixed_spread_bps`` (used only when the simulator is given a top-of-
  book price instead of an L2 snapshot).
* :class:`SimulatedFill` — frozen dataclass with the modeled
  ``fill_quantity``, ``fill_price``, ``slippage_bps`` relative to the
  reference midpoint, and an ``order_state`` (FILLED / PARTIAL / FAILED)
  matching the US-012 lifecycle enum so the rebalance scheduler can feed
  this straight back into ``OrderLifecycle.reconcile_fill(...)`` without
  any translation.
* :func:`simulate_fill_from_book` — pure helper: takes an order + an
  :class:`~src.equity.depth_pricing.L2OrderBook` + a
  :class:`SlippageModel` and returns a :class:`SimulatedFill`. Used by
  the unit-test determinism assertion.
* :func:`simulate_fill_from_quote` — pure helper: takes an order + a
  top-of-book ``reference_price`` and walks the spread / slippage math
  for venues where we only have a midprice quote.
* :class:`ShadowFillSimulator` — armed once for a universe; re-enforces
  the ship gate before every call and writes an atomic JSON ledger of
  every fill it produces so US-017's autonomous loop has an auditable
  record of every shadow trade.

Ship-gate guard
---------------

:func:`enforce_ship_gate` is byte-for-byte identical to the other
equity guards (``src.equity.rebalance._enforce_ship_gate``,
``src.equity.order_lifecycle.enforce_ship_gate``,
``src.equity.kill_switch.enforce_ship_gate``,
``src.brokers.ibkr.enforce_equity_ship_gate``,
``src.equity.depth_pricing.enforce_ship_gate``,
``src.equity.executors.enforce_ship_gate``) so a fixture that passes
one surface passes them all. Constructing the simulator without a
passing gate raises :class:`ShadowFillError`; the gate is re-checked on
every public call so a gate invalidated mid-run fails closed at the
next fill.

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
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from src.equity.depth_pricing import (
    L2OrderBook,
    get_vwap_for_volume,
)
from src.equity.order_lifecycle import EquityOrder, OrderState

logger = logging.getLogger(__name__)


SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
LEDGER_VERSION = 1


class ShadowFillError(RuntimeError):
    """Raised on ship-gate refusal, malformed input, or contract bugs.

    Distinct exception type so the autonomous loop (US-017) and the
    shadow / paper runner (US-021) can match ``except ShadowFillError``
    specifically and refuse to advance their state machines without
    confusing it with broker-side errors.
    """


@dataclass(frozen=True)
class SlippageModel:
    """Deterministic slippage configuration.

    Args:
        extra_slippage_bps: Multiplicative haircut applied to the VWAP
            after the depth walk, in basis points. Models broker
            latency / queue-position / hidden-liquidity effects that the
            visible L2 snapshot cannot see. Default 1.0 bp matches the
            harvester backtest's modeled cost (5 bps round-trip = ~1 bp
            per leg per side after subtracting commission).
        participation_rate: Cap on the fraction of visible depth the
            simulator is allowed to consume in one fill. ``1.0`` means
            "take all visible liquidity"; ``0.25`` means "never take
            more than a quarter of visible depth". Used by US-017 to
            refuse over-trading thin books — the resulting PARTIAL fill
            is the signal that the order should be re-sliced.
        fixed_spread_bps: Half-spread (in bps) used by
            :func:`simulate_fill_from_quote` when the caller only has a
            top-of-book price. Ignored by
            :func:`simulate_fill_from_book` which walks the real
            visible depth.

    All fields are validated at construct time; an invalid model is
    fail-loud (not silently coerced) so callers cannot accidentally
    ship a degenerate slippage configuration.
    """

    extra_slippage_bps: float = 1.0
    participation_rate: float = 1.0
    fixed_spread_bps: float = 1.0

    def __post_init__(self) -> None:
        for name, value, lo, hi in (
            ("extra_slippage_bps", self.extra_slippage_bps, 0.0, 1_000.0),
            ("participation_rate", self.participation_rate, 0.0, 1.0),
            ("fixed_spread_bps", self.fixed_spread_bps, 0.0, 1_000.0),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ShadowFillError(
                    f"SlippageModel.{name} must be a finite number, got {value!r}"
                )
            if value < lo or value > hi:
                raise ShadowFillError(
                    f"SlippageModel.{name} must lie in [{lo}, {hi}], got {value}"
                )
        # participation_rate == 0 would silently zero every fill, which
        # the simulator could not distinguish from "book is empty". Fail
        # loud so the operator picks a real cap.
        if self.participation_rate == 0.0:
            raise ShadowFillError(
                "SlippageModel.participation_rate must be > 0 (got 0); "
                "use a positive cap or omit the field to take all visible depth"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SimulatedFill:
    """Outcome of a shadow fill simulation.

    Args:
        client_order_id: Echoes the
            :class:`~src.equity.order_lifecycle.EquityOrder` field so
            the rebalance scheduler can correlate the fill back to its
            ledger entry without any side state.
        ticker: Symbol traded.
        side: ``"BUY"`` or ``"SELL"``.
        requested_quantity: Whole-share quantity the order asked for.
        fill_quantity: Whole-share quantity the simulator says was
            filled. Less than ``requested_quantity`` for PARTIAL fills.
        fill_price: Modeled fill price after VWAP walk + slippage
            haircut. ``0.0`` when ``fill_quantity == 0`` (could-not-fill
            FAILED case) so JSON ledger readers can grep zero-price rows.
        reference_price: Midprice (L2 case) or supplied reference (quote
            case) the slippage is measured against. Always positive when
            the input was valid.
        slippage_bps: Signed basis points of slippage relative to
            ``reference_price``. Positive == paid more than midprice on
            BUY / received less than midprice on SELL.
        order_state: One of :class:`~src.equity.order_lifecycle.OrderState`
            ``FILLED`` / ``PARTIAL`` / ``FAILED``. Designed to be fed
            straight back into the lifecycle reconcile path.
        reason: Short tag describing why the simulator ended where it
            did (e.g. ``"book_too_thin"``, ``"capped_by_participation"``,
            ``"clean_full_fill"``). Surfaced in the ledger for the
            US-017 supervisor.
    """

    client_order_id: str
    ticker: str
    side: str
    requested_quantity: int
    fill_quantity: int
    fill_price: float
    reference_price: float
    slippage_bps: float
    order_state: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _atomic_write_json(payload: Mapping[str, object], path: Path) -> None:
    """Write ``payload`` to ``path`` via ``mkstemp`` + ``os.replace``.

    Mirrors :func:`src.equity.order_lifecycle._atomic_write_json` and
    :func:`src.equity.executors._atomic_write_json` exactly so a crash
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

    Fail-closed on:

    * file missing
    * file unreadable / not valid JSON
    * payload not a JSON object
    * ``gate_pass`` not exactly ``True``
    * ``universe_hash`` missing / blank / not the expected hash

    Raises:
        ShadowFillError: On any of the above. Identical contract to
            every other equity ship-gate guard.
    """
    path = Path(path)
    if not path.exists():
        raise ShadowFillError(
            f"shadow fills refuses: ship gate not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShadowFillError(
            f"shadow fills refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ShadowFillError(
            f"shadow fills refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise ShadowFillError(
            "shadow fills refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to arm the simulator)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise ShadowFillError(
            f"shadow fills refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise ShadowFillError(
            "shadow fills refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"expected={expected_universe_hash!r}) at {path}"
        )


def _validate_order(order: EquityOrder) -> None:
    if not isinstance(order, EquityOrder):
        raise ShadowFillError(
            f"simulate_fill requires EquityOrder, got {type(order).__name__}"
        )
    if order.side not in ("BUY", "SELL"):
        raise ShadowFillError(
            f"order.side must be 'BUY' or 'SELL', got {order.side!r}"
        )
    if not isinstance(order.quantity, int) or order.quantity <= 0:
        raise ShadowFillError(
            f"order.quantity must be a positive int, got {order.quantity!r}"
        )


def simulate_fill_from_book(
    order: EquityOrder,
    book: L2OrderBook,
    model: SlippageModel,
) -> SimulatedFill:
    """Simulate a fill against a real L2 order book.

    Walks ``book`` with the US-014 VWAP helper to find the size-weighted
    average fill price for ``order.quantity`` shares. Caps the fill at
    ``model.participation_rate * visible_depth`` so the simulator never
    pretends to take more than the operator-supplied share of visible
    liquidity. Applies ``model.extra_slippage_bps`` as a multiplicative
    haircut on the VWAP to model latency / queue-position effects.

    Deterministic: the same (order, book, model) returns the same
    :class:`SimulatedFill` bytewise.
    """
    _validate_order(order)
    if not isinstance(book, L2OrderBook):
        raise ShadowFillError(
            f"simulate_fill_from_book requires L2OrderBook, got "
            f"{type(book).__name__}"
        )
    if not isinstance(model, SlippageModel):
        raise ShadowFillError(
            f"simulate_fill_from_book requires SlippageModel, got "
            f"{type(model).__name__}"
        )

    is_buy = order.side == "BUY"
    mid = book.mid_price()
    if mid is None or mid <= 0.0:
        return SimulatedFill(
            client_order_id=order.client_order_id,
            ticker=order.ticker,
            side=order.side,
            requested_quantity=order.quantity,
            fill_quantity=0,
            fill_price=0.0,
            reference_price=0.0,
            slippage_bps=0.0,
            order_state=OrderState.FAILED.value,
            reason="no_two_sided_book",
        )

    visible_depth = book.total_size(is_buy=is_buy)
    if visible_depth <= 0.0:
        return SimulatedFill(
            client_order_id=order.client_order_id,
            ticker=order.ticker,
            side=order.side,
            requested_quantity=order.quantity,
            fill_quantity=0,
            fill_price=0.0,
            reference_price=float(mid),
            slippage_bps=0.0,
            order_state=OrderState.FAILED.value,
            reason="book_too_thin",
        )

    cap = math.floor(visible_depth * model.participation_rate)
    requested = order.quantity
    fillable = min(requested, cap)
    if fillable <= 0:
        return SimulatedFill(
            client_order_id=order.client_order_id,
            ticker=order.ticker,
            side=order.side,
            requested_quantity=requested,
            fill_quantity=0,
            fill_price=0.0,
            reference_price=float(mid),
            slippage_bps=0.0,
            order_state=OrderState.FAILED.value,
            reason="capped_by_participation_to_zero",
        )

    walk = get_vwap_for_volume(book, is_buy=is_buy, volume=float(fillable))
    walked_quantity = int(math.floor(walk.result_volume))
    if walked_quantity <= 0 or walk.result_price <= 0.0:
        return SimulatedFill(
            client_order_id=order.client_order_id,
            ticker=order.ticker,
            side=order.side,
            requested_quantity=requested,
            fill_quantity=0,
            fill_price=0.0,
            reference_price=float(mid),
            slippage_bps=0.0,
            order_state=OrderState.FAILED.value,
            reason="vwap_walk_empty",
        )
    walked_quantity = min(walked_quantity, fillable)
    vwap = walk.result_price

    haircut = model.extra_slippage_bps / 10_000.0
    if is_buy:
        fill_price = vwap * (1.0 + haircut)
        signed_bps = (fill_price - mid) / mid * 10_000.0
    else:
        fill_price = vwap * (1.0 - haircut)
        signed_bps = (mid - fill_price) / mid * 10_000.0

    if walked_quantity >= requested:
        state = OrderState.FILLED
        reason = "clean_full_fill"
    elif visible_depth >= requested and cap < requested:
        # Cap fired despite the book having enough depth -> participation
        # rate was the binding constraint, not the visible book.
        state = OrderState.PARTIAL
        reason = "capped_by_participation"
    else:
        state = OrderState.PARTIAL
        reason = "book_too_thin"

    return SimulatedFill(
        client_order_id=order.client_order_id,
        ticker=order.ticker,
        side=order.side,
        requested_quantity=requested,
        fill_quantity=int(walked_quantity),
        fill_price=float(fill_price),
        reference_price=float(mid),
        slippage_bps=float(signed_bps),
        order_state=state.value,
        reason=reason,
    )


def simulate_fill_from_quote(
    order: EquityOrder,
    reference_price: float,
    model: SlippageModel,
) -> SimulatedFill:
    """Simulate a fill when only a top-of-book midprice is available.

    Used for symbols where the live L2 snapshot is not yet wired (early
    paper-mode roll-out). The fill price assumes the order crosses the
    half-spread and pays the ``extra_slippage_bps`` haircut on top — so
    a BUY at ``reference_price=$100`` with ``fixed_spread_bps=2`` and
    ``extra_slippage_bps=1`` fills at ``$100 * (1 + 0.0002 + 0.0001)``.
    Deterministic: same inputs yield bytewise identical output.
    """
    _validate_order(order)
    if not isinstance(model, SlippageModel):
        raise ShadowFillError(
            f"simulate_fill_from_quote requires SlippageModel, got "
            f"{type(model).__name__}"
        )
    if (
        not isinstance(reference_price, (int, float))
        or not math.isfinite(reference_price)
        or reference_price <= 0.0
    ):
        raise ShadowFillError(
            f"reference_price must be a positive finite float, "
            f"got {reference_price!r}"
        )

    is_buy = order.side == "BUY"
    requested = order.quantity
    spread_factor = model.fixed_spread_bps / 10_000.0
    haircut_factor = model.extra_slippage_bps / 10_000.0
    total = spread_factor + haircut_factor
    if is_buy:
        fill_price = float(reference_price) * (1.0 + total)
        signed_bps = total * 10_000.0
    else:
        fill_price = float(reference_price) * (1.0 - total)
        signed_bps = total * 10_000.0

    return SimulatedFill(
        client_order_id=order.client_order_id,
        ticker=order.ticker,
        side=order.side,
        requested_quantity=requested,
        fill_quantity=int(requested),
        fill_price=float(fill_price),
        reference_price=float(reference_price),
        slippage_bps=float(signed_bps),
        order_state=OrderState.FILLED.value,
        reason="quote_fixed_spread",
    )


class ShadowFillSimulator:
    """Live shadow-mode simulator: ship-gate guarded, atomic ledger.

    Constructed armed for a specific ``universe_hash``. Every public
    simulation call re-checks ``SHIP_GATE.json`` so an invalidated gate
    mid-session immediately refuses further simulation — same gate
    discipline as the rebalance / kill-switch / order-lifecycle /
    executor surfaces.

    The ledger is an append-only JSON array on disk; every call writes
    the FULL ledger atomically (mkstemp + os.replace) so a crash leaves
    the prior valid file on disk. The simulator caps the ledger at
    ``max_ledger_entries`` and rotates oldest-first when it overflows so
    the file size is bounded.
    """

    def __init__(
        self,
        universe_hash: str,
        ledger_path: Path,
        *,
        ship_gate_path: Optional[Path] = None,
        slippage_model: Optional[SlippageModel] = None,
        max_ledger_entries: int = 10_000,
    ) -> None:
        if not isinstance(universe_hash, str) or not universe_hash:
            raise ShadowFillError(
                f"universe_hash must be a non-empty string, got "
                f"{universe_hash!r}"
            )
        if not isinstance(max_ledger_entries, int) or max_ledger_entries <= 0:
            raise ShadowFillError(
                f"max_ledger_entries must be a positive int, got "
                f"{max_ledger_entries!r}"
            )
        gate_path = Path(
            ship_gate_path
            if ship_gate_path is not None
            else SHIP_GATE_PATH_DEFAULT
        )
        enforce_ship_gate(gate_path, universe_hash)

        self.universe_hash = universe_hash
        self.ledger_path = Path(ledger_path)
        self.ship_gate_path = gate_path
        self.slippage_model = slippage_model or SlippageModel()
        self.max_ledger_entries = max_ledger_entries
        self._entries: List[dict] = []
        self._persist()

    def _reenforce(self) -> None:
        enforce_ship_gate(self.ship_gate_path, self.universe_hash)

    def _persist(self) -> None:
        payload = {
            "version": LEDGER_VERSION,
            "universe_hash": self.universe_hash,
            "slippage_model": self.slippage_model.to_dict(),
            "entries": self._entries,
        }
        _atomic_write_json(payload, self.ledger_path)

    def _record(self, fill: SimulatedFill) -> None:
        entry = fill.to_dict()
        self._entries.append(entry)
        if len(self._entries) > self.max_ledger_entries:
            overflow = len(self._entries) - self.max_ledger_entries
            self._entries = self._entries[overflow:]
        self._persist()

    def fill_from_book(
        self,
        order: EquityOrder,
        book: L2OrderBook,
    ) -> SimulatedFill:
        """Simulate + record a fill against an L2 book."""
        self._reenforce()
        fill = simulate_fill_from_book(order, book, self.slippage_model)
        self._record(fill)
        return fill

    def fill_from_quote(
        self,
        order: EquityOrder,
        reference_price: float,
    ) -> SimulatedFill:
        """Simulate + record a fill against a top-of-book midprice."""
        self._reenforce()
        fill = simulate_fill_from_quote(
            order, reference_price, self.slippage_model
        )
        self._record(fill)
        return fill

    def entries(self) -> List[dict]:
        """Snapshot of recorded ledger entries (defensive copy)."""
        return [dict(entry) for entry in self._entries]


__all__ = [
    "LEDGER_VERSION",
    "SHIP_GATE_PATH_DEFAULT",
    "ShadowFillError",
    "ShadowFillSimulator",
    "SimulatedFill",
    "SlippageModel",
    "enforce_ship_gate",
    "simulate_fill_from_book",
    "simulate_fill_from_quote",
]
