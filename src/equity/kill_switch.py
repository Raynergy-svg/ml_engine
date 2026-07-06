"""US-013 — IBKR-native equity kill switch (``flatten_all``).

The existing ``ExecutionManager.flatten_all`` at
:mod:`src.scanner.execution` is hardcoded to OANDA REST and silently
no-ops against an equity book — equities never sit on OANDA, so its
``pendingOrders``/``openTrades`` calls return empty and the operator
gets a "successful" kill that left every share long. This module
provides an equity-native kill switch that:

* Goes through the :class:`src.brokers.base.BrokerClient` abstraction
  (so it works against IBKR today and any future equity broker without
  touching the kill switch);
* Cancels every working order surfaced by ``broker.get_trades()``
  (which for :class:`~src.brokers.ibkr.IBKRBroker` enumerates open
  IBKR Trade objects — i.e. live working orders);
* Flattens every equity position surfaced by
  ``broker.get_open_positions()`` by placing an opposing whole-share
  market order via ``broker.place_equity_order(...)`` — the same
  audited path the rebalance scheduler uses, so the panic-button can
  never bypass the ship-gate guard;
* Refuses to do anything unless the SHIP_GATE.json artifact still has
  ``gate_pass == True`` AND ``universe_hash`` exactly matches the
  universe the kill switch was armed for. Missing file, malformed
  JSON, false / wrong-hash payload all fail closed.
* Persists every kill event atomically (``tempfile.mkstemp`` +
  ``os.replace``) so a crash mid-write leaves the prior log intact and
  a restarted process can resume from a consistent state.

The class is intentionally synchronous. ``IBKRBroker``'s order and
position calls are sync wrappers around ``ib_async``; ``asyncio`` would
just be ceremony. The OANDA flatten_all is async because every REST
call goes over the wire individually and we needed gather() to overlap
them — IBKR's TWS connection is in-process and serialized internally,
so there's no parallelism to recover.

Surfaces
--------

* :class:`EquityKillSwitch` — armed object that drives the flatten.
* :func:`enforce_ship_gate` — public so tests can drive the guard
  directly. Mirrors :func:`src.brokers.ibkr.enforce_equity_ship_gate`
  and the other equity ship-gate guards so the same fixture passes /
  fails identically across modules.
* :class:`EquityKillSwitchError` — ship-gate refusal / config bug.
* :class:`EquityKillSwitchPartialFailure` — raised after the flatten
  attempt completes if at least one order or position could not be
  closed; carries ``remaining_orders`` and ``remaining_positions`` for
  the supervisor to retry or escalate.
* :class:`KillSwitchResult` — structured outcome the caller can log.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.brokers.base import BrokerClient
from src.brokers.instrument import Instrument
from src.brokers.types import PositionInfo, TradeInfo

logger = logging.getLogger(__name__)

SHIP_GATE_PATH_DEFAULT = "trained_data/backtests/SHIP_GATE.json"
STATE_VERSION = 1

# Re-poll budget for confirming the book is actually flat after flatten
# orders are accepted. A merely-ACCEPTED (PENDING) flatten order is NOT a
# closed position — we must re-read get_open_positions() until the symbol
# is gone from the book or the budget is exhausted. On exhaustion we fail
# closed (escalate) rather than report a flat that isn't confirmed.
CONFIRM_FLAT_MAX_POLLS_DEFAULT = 5
CONFIRM_FLAT_BACKOFF_BASE_S_DEFAULT = 0.5
CONFIRM_FLAT_BACKOFF_MAX_S_DEFAULT = 8.0


class EquityKillSwitchError(RuntimeError):
    """Raised when the kill switch refuses to construct or fire.

    Covers ship-gate failure (missing / malformed / false / mismatched
    universe_hash), bad ``instrument_resolver`` contract, and
    on-disk-state corruption. Distinct from
    :class:`EquityKillSwitchPartialFailure` so the rebalance supervisor
    can match "config bug" separately from "ran but some legs failed".
    """


class EquityKillSwitchPartialFailure(RuntimeError):
    """Raised after ``flatten_all`` completes with un-closed legs.

    The kill attempt itself completed; this signal exists so the
    operator-facing caller (TUI, supervisor) sees a hard error rather
    than mistaking a partial flat for a successful one. Attributes:

    * ``remaining_orders`` — order ids the broker reported it could
      not cancel (also persisted in the event log on disk).
    * ``remaining_positions`` — symbols still long after the flatten.
    """

    def __init__(
        self,
        message: str,
        *,
        remaining_orders: Sequence[str],
        remaining_positions: Sequence[str],
    ) -> None:
        super().__init__(message)
        self.remaining_orders = list(remaining_orders)
        self.remaining_positions = list(remaining_positions)


@dataclass
class KillSwitchResult:
    """Structured outcome of one ``flatten_all`` invocation.

    ``orders_*`` counts come from iterating ``broker.get_trades()``
    (working orders). ``positions_*`` counts come from iterating
    ``broker.get_open_positions()`` filtered to the requested asset
    classes (default ``("EQUITY",)``). ``duration_s`` is wall-clock so
    the supervisor can flag a kill that took implausibly long (e.g.
    broker is hung).
    """

    orders_cancelled: int
    orders_failed: int
    positions_closed: int
    positions_failed: int
    duration_s: float
    remaining_order_ids: List[str] = field(default_factory=list)
    remaining_position_symbols: List[str] = field(default_factory=list)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically: tempfile.mkstemp + os.replace.

    Mirrors :func:`src.equity.order_lifecycle._atomic_write_json` so the
    crash-mid-write semantics match across every equity surface.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".kill_switch_", suffix=".json", dir=str(path.parent),
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
    """Refuse the kill switch unless SHIP_GATE.json clears for THIS universe.

    Behavior is bit-for-bit identical to
    :func:`src.brokers.ibkr.enforce_equity_ship_gate`,
    :func:`src.equity.rebalance._enforce_ship_gate`, and
    :func:`src.equity.order_lifecycle.enforce_ship_gate` so that a
    fixture that passes one surface passes them all. Fail-closed on:

    * file missing
    * file unreadable / not valid JSON
    * payload not a JSON object
    * ``gate_pass`` not exactly ``True``
    * ``universe_hash`` missing / blank / not the expected hash

    Raises:
        EquityKillSwitchError: On any of the above.
    """
    path = Path(path)
    if not path.exists():
        raise EquityKillSwitchError(
            f"equity kill switch refuses: ship gate not found at {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EquityKillSwitchError(
            f"equity kill switch refuses: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise EquityKillSwitchError(
            f"equity kill switch refuses: {path} payload is not a JSON object"
        )
    if payload.get("gate_pass") is not True:
        raise EquityKillSwitchError(
            "equity kill switch refuses: gate_pass="
            f"{payload.get('gate_pass')!r} at {path} "
            "(must be exactly True to arm the kill switch)"
        )
    universe_hash = payload.get("universe_hash")
    if not isinstance(universe_hash, str) or not universe_hash:
        raise EquityKillSwitchError(
            f"equity kill switch refuses: universe_hash missing/blank at {path}"
        )
    if universe_hash != expected_universe_hash:
        raise EquityKillSwitchError(
            "equity kill switch refuses: universe_hash mismatch "
            f"(gate={universe_hash!r}, "
            f"expected={expected_universe_hash!r}) at {path}"
        )


InstrumentResolver = Callable[[str], Optional[Instrument]]


def _opposing_direction(direction: str) -> str:
    """Return the market-flat side for a given position direction.

    ``BrokerClient.get_open_positions`` returns ``"BUY"`` for long
    positions and ``"SELL"`` for short — to flatten we issue the
    opposite side as a market order.
    """
    if direction == "BUY":
        return "SELL"
    if direction == "SELL":
        return "BUY"
    raise EquityKillSwitchError(
        f"unsupported position direction for flatten: {direction!r}"
    )


class EquityKillSwitch:
    """Equity-native, broker-abstraction-routed flatten_all.

    Construction validates the ship gate up-front (so an unarmed kill
    switch object cannot exist). ``flatten_all`` re-checks the gate
    immediately before issuing any broker call — if SHIP_GATE.json was
    invalidated between arming and firing, the kill switch still
    refuses (and logs LOUDLY: an invalidated gate during a kill attempt
    is the worst possible moment to discover the universe drifted).

    The class does not own the broker connection — the caller passes a
    live :class:`BrokerClient` into ``flatten_all``. This keeps the kill
    switch testable against an in-process paper broker without
    duplicating the connect/disconnect lifecycle.
    """

    def __init__(
        self,
        universe_hash: str,
        state_path: Path,
        *,
        ship_gate_path: Optional[Path] = None,
        instrument_resolver: Optional[InstrumentResolver] = None,
        max_event_log_entries: int = 200,
    ) -> None:
        """Arm the kill switch against ``universe_hash``.

        Args:
            universe_hash: Hash of the currently deployed equity
                universe. Must match the persisted ``universe_hash`` in
                SHIP_GATE.json.
            state_path: Path where the kill-switch event log is
                persisted (atomic writes).
            ship_gate_path: Path to SHIP_GATE.json (default:
                ``trained_data/backtests/SHIP_GATE.json``).
            instrument_resolver: Callable mapping ``symbol -> Instrument``
                so the kill switch can build opposing equity orders
                from a :class:`PositionInfo` (which only carries the
                ticker string). Default builds an EQUITY ``Instrument``
                via :meth:`Instrument.equity` — fine for US tickers
                routed via SMART/USD; pass a custom resolver for
                non-US tickers or non-USD denominated names.
            max_event_log_entries: Cap on persisted event-log length —
                older entries are dropped FIFO so the file doesn't grow
                unbounded if the operator hits the panic button daily.

        Raises:
            EquityKillSwitchError: If the ship gate refuses or arguments
                are invalid.
        """
        if not isinstance(universe_hash, str) or not universe_hash:
            raise EquityKillSwitchError(
                "EquityKillSwitch requires a non-empty universe_hash"
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
        self._resolver: InstrumentResolver = (
            instrument_resolver
            if instrument_resolver is not None
            else _default_equity_resolver
        )
        if not callable(self._resolver):
            raise EquityKillSwitchError(
                "instrument_resolver must be callable"
            )
        if max_event_log_entries < 1:
            raise EquityKillSwitchError(
                "max_event_log_entries must be >= 1 "
                f"(got {max_event_log_entries!r})"
            )
        self.max_event_log_entries = int(max_event_log_entries)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def flatten_all(
        self,
        broker: BrokerClient,
        reason: str,
        *,
        asset_classes: Sequence[str] = ("EQUITY",),
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        confirm_flat_max_polls: int = CONFIRM_FLAT_MAX_POLLS_DEFAULT,
        confirm_flat_backoff_base_s: float = (
            CONFIRM_FLAT_BACKOFF_BASE_S_DEFAULT
        ),
        confirm_flat_backoff_max_s: float = (
            CONFIRM_FLAT_BACKOFF_MAX_S_DEFAULT
        ),
    ) -> KillSwitchResult:
        """Cancel every working order and market-flat every equity position.

        Re-validates the ship gate before doing anything. Steps:

        1. Re-enforce the ship gate (fail-closed if invalidated since
           arm).
        2. Cancel every working order returned by
           ``broker.get_trades()`` via ``broker.close_trade(trade_id)``.
           ``close_trade`` returning ``False`` increments
           ``orders_failed``; raising surfaces as an order_failed entry
           with the exception type in the log and continues — one
           uncancellable order must not block the rest.
        3. Pull live positions via ``broker.get_open_positions()``,
           filter by ``asset_classes`` (default ``("EQUITY",)``),
           resolve each symbol to an :class:`Instrument`, and submit a
           whole-share opposing market order via
           ``broker.place_equity_order(...)``. **An accepted order
           (PENDING/FILLED status) is NOT yet a closed position** — it
           only means the broker took the request. The kill switch then
           RE-POLLS ``broker.get_open_positions()`` with bounded retries
           and backoff until every symbol it flattened is gone from the
           book, OR the retry budget is exhausted. A symbol counts as
           CLOSED only when the broker confirms it is absent from the
           live book. A symbol whose flatten order was rejected outright,
           or which is still present after the budget is exhausted,
           counts as FAILED (remaining) — this is the fail-closed path:
           a kill switch must never declare flat while exposure may be
           live.
        4. Persist a structured event to ``state_path`` atomically.
        5. Raise :class:`EquityKillSwitchPartialFailure` if any leg
           failed; the caller is the supervisor and MUST see a hard
           error rather than a green light.

        Args:
            broker: Live :class:`BrokerClient` (already connected).
            reason: Human-readable reason for the audit log.
            asset_classes: Asset classes to flatten. Defaults to just
                EQUITY; the FX/futures kill paths live elsewhere and
                this method intentionally refuses to flatten them so
                callers cannot accidentally cross-wire the equity panic
                button into the FX book.
            now_fn: UTC clock injection point (kept simple for tests
                that pin a deterministic timestamp without monkey-
                patching ``datetime``).
            monotonic_fn: Wall-clock duration injection point.
            sleep_fn: Backoff sleep injection point. Tests pass a no-op
                so the bounded re-poll loop runs instantly without real
                wall-clock delay; production uses :func:`time.sleep`.
            confirm_flat_max_polls: Number of times to re-poll
                ``get_open_positions()`` to confirm the book is flat
                AFTER sending flatten orders. Must be >= 1. The first
                poll happens immediately (no sleep); subsequent polls
                back off. If positions remain after the last poll, they
                are reported as remaining (escalation).
            confirm_flat_backoff_base_s: Base backoff seconds between
                re-polls (exponential, doubling each attempt).
            confirm_flat_backoff_max_s: Backoff ceiling per sleep.

        Returns:
            :class:`KillSwitchResult` even when partial — the exception
            attaches the same data so the caller can introspect either
            way.

        Raises:
            EquityKillSwitchError: If the ship gate was invalidated
                between arming and firing, or if the broker contract is
                violated (e.g. resolver returns a non-EQUITY
                Instrument).
            EquityKillSwitchPartialFailure: If any order failed to
                cancel, or any position could not be CONFIRMED flat
                within the re-poll budget. An accepted-but-unconfirmed
                flatten is a FAILURE, not a success.
        """
        if not isinstance(reason, str) or not reason:
            raise EquityKillSwitchError(
                "flatten_all requires a non-empty reason for the audit log"
            )
        asset_classes_set = {str(ac) for ac in asset_classes}
        if not asset_classes_set:
            raise EquityKillSwitchError(
                "flatten_all requires at least one asset_class to flatten"
            )
        if confirm_flat_max_polls < 1:
            raise EquityKillSwitchError(
                "confirm_flat_max_polls must be >= 1 "
                f"(got {confirm_flat_max_polls!r})"
            )

        # Step 1: re-enforce the gate. Don't trust the constructor's
        # check — the universe could have been re-deployed since.
        enforce_ship_gate(self.ship_gate_path, self.universe_hash)

        start = monotonic_fn()
        logger.warning(
            "equity kill switch firing: reason=%s asset_classes=%s",
            reason, sorted(asset_classes_set),
        )

        # Step 2: cancel working orders.
        orders_cancelled, orders_failed, remaining_orders = (
            self._cancel_working_orders(broker)
        )

        # Step 3: flatten positions, then RE-POLL to confirm flat.
        (
            positions_closed,
            positions_failed,
            remaining_positions,
        ) = self._flatten_positions(
            broker,
            asset_classes_set,
            sleep_fn=sleep_fn,
            max_polls=confirm_flat_max_polls,
            backoff_base_s=confirm_flat_backoff_base_s,
            backoff_max_s=confirm_flat_backoff_max_s,
        )

        duration_s = monotonic_fn() - start
        result = KillSwitchResult(
            orders_cancelled=orders_cancelled,
            orders_failed=orders_failed,
            positions_closed=positions_closed,
            positions_failed=positions_failed,
            duration_s=round(duration_s, 4),
            remaining_order_ids=remaining_orders,
            remaining_position_symbols=remaining_positions,
        )

        # Step 4: persist event.
        self._append_event(reason, result, now_fn())

        # Step 5: hard-fail on partial.
        if remaining_orders or remaining_positions:
            raise EquityKillSwitchPartialFailure(
                (
                    f"equity kill switch partial: "
                    f"{orders_failed} order(s) and "
                    f"{positions_failed} position(s) still open "
                    f"(remaining_orders={remaining_orders}, "
                    f"remaining_positions={remaining_positions})"
                ),
                remaining_orders=remaining_orders,
                remaining_positions=remaining_positions,
            )

        logger.info(
            "equity kill switch complete: result=%s", asdict(result),
        )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cancel_working_orders(
        self, broker: BrokerClient
    ) -> tuple[int, int, List[str]]:
        try:
            trades: List[TradeInfo] = list(broker.get_trades())
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.error(
                "kill switch: get_trades() failed: %s", exc, exc_info=True
            )
            # Re-raise: we cannot proceed if the broker can't enumerate
            # working orders. Failing closed beats canceling nothing
            # and reporting success.
            raise EquityKillSwitchError(
                f"kill switch could not enumerate working orders: {exc}"
            ) from exc

        cancelled = 0
        failed = 0
        remaining: List[str] = []
        for trade in trades:
            trade_id = str(trade.trade_id)
            if not trade_id:
                logger.warning(
                    "kill switch: skipping working order with blank trade_id"
                )
                continue
            ok = False
            try:
                ok = bool(broker.close_trade(trade_id))
            except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
                logger.error(
                    "kill switch: close_trade(%s) raised %s: %s",
                    trade_id, type(exc).__name__, exc,
                )
            if ok:
                cancelled += 1
                logger.info("kill switch: cancelled order %s", trade_id)
            else:
                failed += 1
                remaining.append(trade_id)
                logger.warning(
                    "kill switch: failed to cancel order %s", trade_id
                )
        return cancelled, failed, remaining

    def _flatten_positions(
        self,
        broker: BrokerClient,
        asset_classes: set,
        *,
        sleep_fn: Callable[[float], None],
        max_polls: int,
        backoff_base_s: float,
        backoff_max_s: float,
    ) -> tuple[int, int, List[str]]:
        """Send opposing flats, then re-poll until the book confirms flat.

        Fail-closed contract: a position is CLOSED only when the broker's
        ``get_open_positions()`` no longer reports it. An accepted
        (PENDING/FILLED) flatten order is necessary but NOT sufficient —
        we re-poll the live book with bounded backoff and only credit a
        close once the symbol is gone. Symbols whose order was rejected,
        or which remain after the retry budget is exhausted, are reported
        as remaining so :meth:`flatten_all` escalates.
        """
        positions = self._enumerate_positions(broker)

        # Symbols we attempted to flatten and the broker ACCEPTED — these
        # must be CONFIRMED gone from the book before we count them closed.
        awaiting_confirm: set = set()
        # Symbols that failed up-front (rejected order / un-flattenable
        # quantity). These never enter the confirm loop.
        failed_symbols: List[str] = []

        for position in positions:
            symbol = str(position.instrument)
            quantity = position.quantity
            direction = str(position.direction)
            instrument = self._resolver(symbol)
            if instrument is None:
                # Resolver could not classify — log and skip. Skipping
                # is correct: the equity kill switch only flattens
                # equity, and an unresolvable symbol could be FX /
                # futures whose flatten path lives elsewhere.
                logger.info(
                    "kill switch: resolver returned None for %s — "
                    "skipping (non-equity or unknown ticker)",
                    symbol,
                )
                continue
            if instrument.asset_class not in asset_classes:
                logger.info(
                    "kill switch: %s asset_class=%s not in %s — skipping",
                    symbol, instrument.asset_class, sorted(asset_classes),
                )
                continue
            if quantity <= 0:
                logger.info(
                    "kill switch: position %s has quantity=%s — nothing to do",
                    symbol, quantity,
                )
                continue
            try:
                flat_qty = int(quantity)
            except (TypeError, ValueError) as exc:
                logger.error(
                    "kill switch: position %s quantity not convertible "
                    "to int (%r): %s — counting as failed",
                    symbol, quantity, exc,
                )
                failed_symbols.append(symbol)
                continue
            if flat_qty <= 0:
                logger.info(
                    "kill switch: position %s flat quantity rounds to 0 "
                    "(raw=%s) — nothing to do",
                    symbol, quantity,
                )
                continue

            opposing = _opposing_direction(direction)
            accepted = self._place_flat_order(
                broker, instrument, opposing, flat_qty,
            )
            if accepted:
                # Order ACCEPTED is NOT confirmation of flat — defer the
                # close credit to the re-poll loop below.
                awaiting_confirm.add(symbol)
                logger.info(
                    "kill switch: flat order accepted for %s "
                    "(%s %d shares) — awaiting flat confirmation",
                    symbol, opposing, flat_qty,
                )
            else:
                failed_symbols.append(symbol)
                logger.error(
                    "kill switch: flat order REJECTED for %s "
                    "(%s %d shares) — position still open",
                    symbol, opposing, flat_qty,
                )

        # Re-poll the live book to CONFIRM the accepted flats actually
        # cleared. Until proven flat, exposure is assumed live.
        confirmed_flat = self._confirm_flat(
            broker,
            awaiting_confirm,
            sleep_fn=sleep_fn,
            max_polls=max_polls,
            backoff_base_s=backoff_base_s,
            backoff_max_s=backoff_max_s,
        )

        still_open = sorted(awaiting_confirm - confirmed_flat)
        remaining = sorted(set(failed_symbols) | set(still_open))
        closed = len(confirmed_flat)
        failed = len(remaining)
        return closed, failed, remaining

    def _enumerate_positions(
        self, broker: BrokerClient
    ) -> List[PositionInfo]:
        """Read live positions, failing closed if the broker can't answer."""
        try:
            return list(broker.get_open_positions())
        except (ConnectionError, OSError, TimeoutError) as exc:
            logger.error(
                "kill switch: get_open_positions() failed: %s",
                exc, exc_info=True,
            )
            raise EquityKillSwitchError(
                f"kill switch could not enumerate open positions: {exc}"
            ) from exc

    def _confirm_flat(
        self,
        broker: BrokerClient,
        awaiting_confirm: set,
        *,
        sleep_fn: Callable[[float], None],
        max_polls: int,
        backoff_base_s: float,
        backoff_max_s: float,
    ) -> set:
        """Re-poll the live book until ``awaiting_confirm`` symbols are flat.

        Returns the set of symbols CONFIRMED absent from
        ``get_open_positions()`` (i.e. actually flat). Symbols still
        present after ``max_polls`` are NOT in the returned set — the
        caller treats them as remaining/failed and escalates.

        The first poll runs immediately; subsequent polls sleep with
        exponential backoff (capped). Each poll logs the attempt number
        and the symbols still open so the operator can watch the book
        drain (or fail to).
        """
        confirmed: set = set()
        if not awaiting_confirm:
            return confirmed

        for attempt in range(1, max_polls + 1):
            try:
                live = list(broker.get_open_positions())
            except (ConnectionError, OSError, TimeoutError) as exc:
                # Cannot confirm flat — fail closed. Do NOT credit closes
                # we could not verify; surface as a hard error so a human
                # checks the book manually.
                logger.error(
                    "kill switch: re-poll get_open_positions() failed on "
                    "attempt %d/%d: %s", attempt, max_polls, exc,
                    exc_info=True,
                )
                raise EquityKillSwitchError(
                    "kill switch could not confirm flat: re-poll failed "
                    f"after {attempt - 1} confirmed poll(s): {exc}"
                ) from exc

            open_symbols = {str(p.instrument) for p in live}
            newly_flat = {
                s for s in awaiting_confirm
                if s not in confirmed and s not in open_symbols
            }
            confirmed |= newly_flat

            still_open = sorted(awaiting_confirm - confirmed)
            if not still_open:
                logger.info(
                    "kill switch: book confirmed flat on attempt %d/%d "
                    "(%d position(s) closed)",
                    attempt, max_polls, len(confirmed),
                )
                return confirmed

            logger.warning(
                "kill switch: re-poll %d/%d — %d position(s) still open: %s",
                attempt, max_polls, len(still_open), still_open,
            )
            if attempt < max_polls:
                delay = min(
                    backoff_base_s * (2 ** (attempt - 1)), backoff_max_s
                )
                sleep_fn(delay)

        # Budget exhausted with positions still on the book.
        unconfirmed = sorted(awaiting_confirm - confirmed)
        logger.error(
            "kill switch: re-poll budget exhausted (%d polls) — %d "
            "position(s) NOT confirmed flat: %s — escalating",
            max_polls, len(unconfirmed), unconfirmed,
        )
        return confirmed

    def _place_flat_order(
        self,
        broker: BrokerClient,
        instrument: Instrument,
        direction: str,
        quantity: int,
    ) -> bool:
        """Submit one whole-share opposing market order.

        Returns ``True`` if the broker ACCEPTED the order (status in
        ``{PENDING, FILLED}``), ``False`` if it rejected or raised a
        recoverable broker error. **Acceptance is NOT confirmation that
        the position closed** — the caller must re-poll
        ``get_open_positions()`` to confirm flat. This method only
        distinguishes "request taken" from "request refused".

        ``BrokerClient`` does not enshrine ``place_equity_order`` in
        the ABC (that method lives on :class:`IBKRBroker` for now), so
        we look it up via ``getattr`` and refuse loudly if missing.
        """
        place_equity_order = getattr(broker, "place_equity_order", None)
        if place_equity_order is None or not callable(place_equity_order):
            raise EquityKillSwitchError(
                "kill switch broker contract violated: "
                f"{type(broker).__name__} does not expose "
                "place_equity_order(instrument, direction, quantity)"
            )
        try:
            result = place_equity_order(instrument, direction, quantity)
        except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
            logger.error(
                "kill switch: place_equity_order raised %s: %s",
                type(exc).__name__, exc,
            )
            return False
        status = getattr(result, "status", None)
        # PENDING == broker accepted the order; FILLED == synchronous
        # broker confirmed the fill in-band. Either means the request was
        # taken. Anything else (e.g. "rejected") means the broker refused
        # the order outright — that symbol cannot enter the confirm loop.
        # NOTE: acceptance != flat. The caller re-polls get_open_positions
        # to confirm the position actually cleared before crediting it.
        return str(status).upper() in {"PENDING", "FILLED"}

    def _append_event(
        self,
        reason: str,
        result: KillSwitchResult,
        when: datetime,
    ) -> None:
        """Persist the kill event to ``state_path`` atomically.

        State schema (versioned):
            {
              "version": 1,
              "universe_hash": "...",
              "events": [
                  {"ts": iso, "reason": str, "result": {...}},
                  ...
              ]
            }

        Older events are dropped FIFO past ``max_event_log_entries``.
        Corrupted prior state is treated as missing — the kill event
        itself MUST be recorded; refusing to write because a stale log
        is unparseable would lose audit evidence.
        """
        events: List[Dict[str, Any]] = []
        if self.state_path.exists():
            try:
                raw = self.state_path.read_text(encoding="utf-8")
                payload = json.loads(raw)
                if (
                    isinstance(payload, Mapping)
                    and payload.get("universe_hash") == self.universe_hash
                    and isinstance(payload.get("events"), list)
                ):
                    for item in payload["events"]:
                        if isinstance(item, Mapping):
                            events.append(dict(item))
                else:
                    logger.warning(
                        "kill switch: prior state at %s has mismatched "
                        "schema or universe_hash; starting fresh log",
                        self.state_path,
                    )
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "kill switch: cannot load prior state at %s (%s); "
                    "starting fresh log", self.state_path, exc,
                )
        events.append(
            {
                "ts": when.isoformat(),
                "reason": reason,
                "result": asdict(result),
            }
        )
        if len(events) > self.max_event_log_entries:
            events = events[-self.max_event_log_entries:]
        payload = {
            "version": STATE_VERSION,
            "universe_hash": self.universe_hash,
            "events": events,
        }
        _atomic_write_json(self.state_path, payload)


def _default_equity_resolver(symbol: str) -> Optional[Instrument]:
    """Default symbol -> Instrument resolver for the kill switch.

    Builds an EQUITY :class:`Instrument` via :meth:`Instrument.equity`.
    Returns ``None`` on any construction error (e.g. blank symbol) so
    the kill switch logs+skips rather than aborting the whole flatten
    on one unparseable PositionInfo.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    try:
        return Instrument.equity(symbol.strip().upper())
    except (ValueError, TypeError) as exc:
        logger.warning(
            "default equity resolver: cannot build Instrument for %r: %s",
            symbol, exc,
        )
        return None


__all__ = [
    "EquityKillSwitch",
    "EquityKillSwitchError",
    "EquityKillSwitchPartialFailure",
    "KillSwitchResult",
    "SHIP_GATE_PATH_DEFAULT",
    "STATE_VERSION",
    "enforce_ship_gate",
]
