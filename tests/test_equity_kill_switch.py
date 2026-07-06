"""US-013 — IBKR-native equity kill switch (``flatten_all``) tests.

No mocks. The test broker is a real concrete subclass of
:class:`src.brokers.base.BrokerClient` — a fully functional in-process
paper broker (not a :mod:`unittest.mock` test double). It tracks
positions and working orders in plain Python state so the kill switch
runs end-to-end against a real :class:`BrokerClient` interface.

Acceptance criteria covered:

* :func:`enforce_ship_gate` rejects missing / unreadable / non-dict /
  ``gate_pass != True`` / missing-hash / mismatched-hash payloads.
* :class:`EquityKillSwitch` constructor fails closed against every
  bad-gate scenario.
* ``flatten_all`` cancels working orders via
  :meth:`BrokerClient.close_trade` (real call against the in-process
  broker).
* ``flatten_all`` flattens equity positions via
  ``broker.place_equity_order(...)`` with the opposing side and
  ``int(quantity)`` shares.
* ``flatten_all`` re-enforces the ship gate at fire-time (gate
  invalidated between arm + fire is rejected even though the object
  exists).
* Partial failures raise :class:`EquityKillSwitchPartialFailure` and
  the result still records what DID get cancelled / closed.
* Persisted event log on disk is JSON, atomic, and version-stamped;
  FIFO eviction caps the log length.
* Integration test against a real paper IBGateway is marked
  ``@pytest.mark.integration`` and skips cleanly when the gateway is
  unreachable.
* ``src/equity/kill_switch.py`` contains zero bare ``except:`` clauses
  (greps the on-disk source).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from src.brokers.base import BrokerClient
from src.brokers.instrument import Instrument
from src.brokers.types import (
    AccountSummary,
    OrderResult,
    PositionInfo,
    TradeInfo,
)
from src.equity.kill_switch import (
    SHIP_GATE_PATH_DEFAULT,
    STATE_VERSION,
    EquityKillSwitch,
    EquityKillSwitchError,
    EquityKillSwitchPartialFailure,
    KillSwitchResult,
    enforce_ship_gate,
)


VALID_HASH = "8bfe419a0c1c06306ff3fa35c39132df522b17ae31718a1dd5463097b467899c"


# ---------------------------------------------------------------------------
# Real-disk fixtures
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".killswitch_test_", suffix=".json", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _passing_payload(universe_hash: str = VALID_HASH, *, hash_: Optional[str] = None) -> dict:
    if hash_ is not None:
        universe_hash = hash_
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


# ---------------------------------------------------------------------------
# Real in-process paper broker
# ---------------------------------------------------------------------------


class PaperEquityBroker(BrokerClient):
    """Concrete BrokerClient that simulates equity paper trading in-process.

    Real subclass of :class:`BrokerClient` (NOT a mock). Tracks working
    orders and open positions in plain Python state; supports the four
    methods the kill switch calls (``get_trades``, ``close_trade``,
    ``get_open_positions``, ``place_equity_order``) plus enough of the
    abstract interface to satisfy the ABC.

    Configurable failure injection (no mocks — plain flags):

    * ``orders_to_fail_close`` — set of trade_ids whose ``close_trade``
      returns ``False`` (simulating a cancel that didn't take).
    * ``positions_to_fail_flat`` — set of symbols whose
      ``place_equity_order`` returns a ``rejected`` OrderResult.
    * ``get_trades_raises`` / ``get_positions_raises`` — Exception
      instances to raise from those methods (simulating broker
      enumeration failure).
    """

    def __init__(self) -> None:
        self._working_orders: Dict[str, TradeInfo] = {}
        self._positions: Dict[str, PositionInfo] = {}
        self._next_id = 100
        self.placed_orders: List[Dict[str, object]] = []
        self.cancel_calls: List[str] = []
        self.orders_to_fail_close: set = set()
        self.positions_to_fail_flat: set = set()
        self.get_trades_raises: Optional[Exception] = None
        self.get_positions_raises: Optional[Exception] = None

    # -- Test helpers (not part of the BrokerClient ABC) ---------------

    def seed_position(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        avg_price: float = 100.0,
    ) -> None:
        self._positions[symbol] = PositionInfo(
            instrument=symbol,
            direction=direction,
            quantity=quantity,
            avg_price=avg_price,
            unrealized_pnl=0.0,
        )

    def seed_working_order(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        quantity: float,
    ) -> None:
        self._working_orders[trade_id] = TradeInfo(
            trade_id=trade_id,
            instrument=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=0.0,
            sl_price=None,
            tp_price=None,
        )

    # -- BrokerClient ABC: required full surface -----------------------

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def fetch_candles(self, instrument, granularity, count):  # pragma: no cover
        raise NotImplementedError("PaperEquityBroker does not serve candles")

    def place_order(  # pragma: no cover - FX bracket not exercised here
        self, instrument, direction, quantity, sl_price, tp_price, entry_price
    ):
        raise NotImplementedError("PaperEquityBroker is equity-only")

    def close_trade(self, trade_id: str) -> bool:
        self.cancel_calls.append(trade_id)
        if trade_id in self.orders_to_fail_close:
            return False
        existed = trade_id in self._working_orders
        self._working_orders.pop(trade_id, None)
        return existed

    def get_nav(self) -> float:
        return 1_000_000.0

    def get_open_positions(self) -> List[PositionInfo]:
        if self.get_positions_raises is not None:
            raise self.get_positions_raises
        return list(self._positions.values())

    def get_price_quote(self, instrument):
        return {"bid": 100.0, "ask": 100.1}

    def get_trades(self) -> List[TradeInfo]:
        if self.get_trades_raises is not None:
            raise self.get_trades_raises
        return list(self._working_orders.values())

    def get_account_summary(self) -> AccountSummary:
        return AccountSummary(
            nav=1_000_000.0,
            balance=1_000_000.0,
            unrealized_pnl=0.0,
            margin_used=0.0,
        )

    # -- Equity-specific (mirrors IBKRBroker.place_equity_order) -------

    def place_equity_order(
        self,
        instrument: Instrument,
        direction: str,
        quantity: int,
    ) -> OrderResult:
        symbol = instrument.symbol
        self.placed_orders.append(
            {
                "symbol": symbol,
                "direction": direction,
                "quantity": quantity,
                "asset_class": instrument.asset_class,
            }
        )
        if symbol in self.positions_to_fail_flat:
            return OrderResult(
                trade_id="0",
                fill_price=0.0,
                status="rejected",
                error_message="paper broker forced rejection",
            )
        # Apply the fill to in-process state so a subsequent
        # get_open_positions() reflects the flat.
        existing = self._positions.get(symbol)
        if existing is not None:
            net_qty = existing.quantity
            net_qty -= quantity if direction != existing.direction else 0
            if net_qty <= 0:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = PositionInfo(
                    instrument=symbol,
                    direction=existing.direction,
                    quantity=net_qty,
                    avg_price=existing.avg_price,
                    unrealized_pnl=existing.unrealized_pnl,
                )
        self._next_id += 1
        return OrderResult(
            trade_id=str(self._next_id),
            fill_price=0.0,
            status="PENDING",
        )


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _make_kill_switch(tmp_path: Path, **kwargs) -> EquityKillSwitch:
    gate = kwargs.pop("ship_gate_path", None) or _ship_gate(tmp_path)
    return EquityKillSwitch(
        universe_hash=VALID_HASH,
        state_path=tmp_path / "kill_switch_state.json",
        ship_gate_path=gate,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Ship-gate guard
# ---------------------------------------------------------------------------


class TestShipGateGuard:
    def test_passes_with_matching_hash_and_gate_pass_true(self, tmp_path):
        gate = _ship_gate(tmp_path)
        enforce_ship_gate(gate, VALID_HASH)  # does not raise

    def test_missing_file_refuses(self, tmp_path):
        missing = tmp_path / "absent.json"
        with pytest.raises(EquityKillSwitchError, match="not found"):
            enforce_ship_gate(missing, VALID_HASH)

    def test_unreadable_json_refuses(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        path.write_text("not valid json", encoding="utf-8")
        with pytest.raises(EquityKillSwitchError, match="cannot read"):
            enforce_ship_gate(path, VALID_HASH)

    def test_non_object_payload_refuses(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(EquityKillSwitchError, match="not a JSON object"):
            enforce_ship_gate(path, VALID_HASH)

    def test_gate_pass_false_refuses(self, tmp_path):
        payload = _passing_payload()
        payload["gate_pass"] = False
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, payload)
        with pytest.raises(EquityKillSwitchError, match="gate_pass="):
            enforce_ship_gate(path, VALID_HASH)

    def test_gate_pass_truthy_not_true_refuses(self, tmp_path):
        # gate_pass MUST be exactly True — string "true" / 1 / "yes" all refuse.
        for truthy in ("true", 1, "yes", ["true"]):
            payload = _passing_payload()
            payload["gate_pass"] = truthy
            path = tmp_path / "SHIP_GATE.json"
            _atomic_write_json(path, payload)
            with pytest.raises(EquityKillSwitchError, match="gate_pass="):
                enforce_ship_gate(path, VALID_HASH)

    def test_missing_universe_hash_refuses(self, tmp_path):
        payload = _passing_payload()
        payload.pop("universe_hash")
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, payload)
        with pytest.raises(
            EquityKillSwitchError, match="universe_hash missing/blank"
        ):
            enforce_ship_gate(path, VALID_HASH)

    def test_blank_universe_hash_refuses(self, tmp_path):
        payload = _passing_payload(hash_="")
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, payload)
        with pytest.raises(
            EquityKillSwitchError, match="universe_hash missing/blank"
        ):
            enforce_ship_gate(path, VALID_HASH)

    def test_mismatched_universe_hash_refuses(self, tmp_path):
        path = _ship_gate(tmp_path, hash_="b" * 64)
        with pytest.raises(EquityKillSwitchError, match="universe_hash mismatch"):
            enforce_ship_gate(path, VALID_HASH)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_constructs_with_valid_gate(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        assert ks.universe_hash == VALID_HASH
        assert ks.ship_gate_path.exists()

    def test_blank_hash_refuses(self, tmp_path):
        with pytest.raises(EquityKillSwitchError, match="non-empty universe_hash"):
            EquityKillSwitch(
                universe_hash="",
                state_path=tmp_path / "s.json",
                ship_gate_path=_ship_gate(tmp_path),
            )

    def test_constructor_refuses_when_gate_missing(self, tmp_path):
        with pytest.raises(EquityKillSwitchError, match="ship gate not found"):
            EquityKillSwitch(
                universe_hash=VALID_HASH,
                state_path=tmp_path / "s.json",
                ship_gate_path=tmp_path / "no_such_gate.json",
            )

    def test_constructor_refuses_when_gate_fails(self, tmp_path):
        payload = _passing_payload()
        payload["gate_pass"] = False
        gate = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(gate, payload)
        with pytest.raises(EquityKillSwitchError, match="gate_pass="):
            EquityKillSwitch(
                universe_hash=VALID_HASH,
                state_path=tmp_path / "s.json",
                ship_gate_path=gate,
            )

    def test_constructor_refuses_on_hash_mismatch(self, tmp_path):
        gate = _ship_gate(tmp_path, hash_="c" * 64)
        with pytest.raises(EquityKillSwitchError, match="universe_hash mismatch"):
            EquityKillSwitch(
                universe_hash=VALID_HASH,
                state_path=tmp_path / "s.json",
                ship_gate_path=gate,
            )

    def test_non_callable_resolver_refuses(self, tmp_path):
        with pytest.raises(EquityKillSwitchError, match="must be callable"):
            EquityKillSwitch(
                universe_hash=VALID_HASH,
                state_path=tmp_path / "s.json",
                ship_gate_path=_ship_gate(tmp_path),
                instrument_resolver="not_callable",  # type: ignore[arg-type]
            )

    def test_zero_event_log_cap_refuses(self, tmp_path):
        with pytest.raises(EquityKillSwitchError, match="max_event_log_entries"):
            EquityKillSwitch(
                universe_hash=VALID_HASH,
                state_path=tmp_path / "s.json",
                ship_gate_path=_ship_gate(tmp_path),
                max_event_log_entries=0,
            )

    def test_default_path_constant_matches_canonical(self):
        assert SHIP_GATE_PATH_DEFAULT == "trained_data/backtests/SHIP_GATE.json"


# ---------------------------------------------------------------------------
# flatten_all happy path
# ---------------------------------------------------------------------------


class TestFlattenAllHappyPath:
    def test_clean_book_succeeds_with_zero_counts(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        result = ks.flatten_all(broker, reason="unit_test")
        assert isinstance(result, KillSwitchResult)
        assert result.orders_cancelled == 0
        assert result.orders_failed == 0
        assert result.positions_closed == 0
        assert result.positions_failed == 0
        assert result.remaining_order_ids == []
        assert result.remaining_position_symbols == []
        assert result.duration_s >= 0.0

    def test_cancels_all_working_orders(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.seed_working_order("11", "AAPL", "BUY", 10)
        broker.seed_working_order("12", "MSFT", "SELL", 20)
        broker.seed_working_order("13", "GOOGL", "BUY", 5)

        result = ks.flatten_all(broker, reason="unit_test")
        assert result.orders_cancelled == 3
        assert result.orders_failed == 0
        assert sorted(broker.cancel_calls) == ["11", "12", "13"]
        # Working orders should be drained from the broker.
        assert broker.get_trades() == []

    def test_flattens_long_equity_positions_with_opposing_sell(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.seed_position("AAPL", "BUY", 50)
        broker.seed_position("MSFT", "BUY", 25)

        result = ks.flatten_all(broker, reason="unit_test")
        assert result.positions_closed == 2
        assert result.positions_failed == 0

        symbols = sorted(o["symbol"] for o in broker.placed_orders)
        assert symbols == ["AAPL", "MSFT"]
        for placed in broker.placed_orders:
            assert placed["direction"] == "SELL"
            assert placed["asset_class"] == "EQUITY"
            assert isinstance(placed["quantity"], int)

        # Broker state must reflect the flat.
        assert broker.get_open_positions() == []

    def test_flattens_short_equity_positions_with_opposing_buy(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.seed_position("AAPL", "SELL", 30)

        result = ks.flatten_all(broker, reason="unit_test")
        assert result.positions_closed == 1
        assert broker.placed_orders[0]["direction"] == "BUY"
        assert broker.placed_orders[0]["quantity"] == 30

    def test_quantity_floored_to_int_for_whole_shares(self, tmp_path):
        # IBKR PositionInfo can carry a float quantity for fractional
        # accounts; the kill switch must round DOWN to whole shares for
        # the ORDER. A whole-share flatten of a 12.7-share long leaves a
        # 0.7-share residual, so the book is NOT confirmed flat — the
        # kill switch must escalate (M2: confirm flat, do not assume it).
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.seed_position("AAPL", "BUY", 12.7)

        with pytest.raises(EquityKillSwitchPartialFailure) as exc_info:
            ks.flatten_all(
                broker, reason="unit_test", sleep_fn=lambda _d: None,
            )
        # The order quantity was floored to whole shares...
        assert broker.placed_orders[0]["quantity"] == 12
        # ...but the 0.7-share residual means the book is not confirmed
        # flat, so the symbol is surfaced rather than falsely credited.
        assert exc_info.value.remaining_positions == ["AAPL"]

    def test_cancels_orders_and_flattens_positions_together(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.seed_working_order("99", "AAPL", "BUY", 10)
        broker.seed_position("AAPL", "BUY", 50)
        broker.seed_position("MSFT", "BUY", 5)

        result = ks.flatten_all(broker, reason="combined")
        assert result.orders_cancelled == 1
        assert result.positions_closed == 2
        assert result.orders_failed == 0
        assert result.positions_failed == 0


# ---------------------------------------------------------------------------
# flatten_all guard re-check and skips
# ---------------------------------------------------------------------------


class TestFlattenAllGuardsAndSkips:
    def test_refuses_at_fire_time_if_gate_invalidated(self, tmp_path):
        gate = _ship_gate(tmp_path)
        ks = EquityKillSwitch(
            universe_hash=VALID_HASH,
            state_path=tmp_path / "s.json",
            ship_gate_path=gate,
        )
        # Operator re-deploys a different universe AFTER arm. Kill
        # switch MUST refuse, not silently fire against stale state.
        invalidated = _passing_payload()
        invalidated["gate_pass"] = False
        _atomic_write_json(gate, invalidated)
        broker = PaperEquityBroker()
        with pytest.raises(EquityKillSwitchError, match="gate_pass="):
            ks.flatten_all(broker, reason="invalidated")

    def test_refuses_blank_reason(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        with pytest.raises(EquityKillSwitchError, match="non-empty reason"):
            ks.flatten_all(PaperEquityBroker(), reason="")

    def test_refuses_empty_asset_classes(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        with pytest.raises(
            EquityKillSwitchError, match="at least one asset_class"
        ):
            ks.flatten_all(
                PaperEquityBroker(), reason="x", asset_classes=()
            )

    def test_skips_non_equity_positions_via_default_resolver(self, tmp_path):
        # The default resolver always builds an EQUITY Instrument,
        # so a custom resolver that returns FX must be honored: the
        # FX position should be SKIPPED, not flattened by the equity
        # kill switch.
        def fx_resolver(symbol: str) -> Optional[Instrument]:
            if symbol == "EUR_USD":
                return Instrument.fx(
                    symbol="EUR_USD",
                    broker_symbol="EUR/USD",
                    pip_value=0.0001,
                )
            return Instrument.equity(symbol)

        ks = _make_kill_switch(tmp_path, instrument_resolver=fx_resolver)
        broker = PaperEquityBroker()
        broker.seed_position("AAPL", "BUY", 10)
        broker.seed_position("EUR_USD", "BUY", 100_000)

        result = ks.flatten_all(broker, reason="mixed_book")
        # Only AAPL gets flattened. EUR_USD is left alone (other kill
        # path owns FX).
        assert result.positions_closed == 1
        assert len(broker.placed_orders) == 1
        assert broker.placed_orders[0]["symbol"] == "AAPL"

    def test_skips_position_when_resolver_returns_none(self, tmp_path):
        def resolver(_symbol: str) -> Optional[Instrument]:
            return None

        ks = _make_kill_switch(tmp_path, instrument_resolver=resolver)
        broker = PaperEquityBroker()
        broker.seed_position("AAPL", "BUY", 10)
        result = ks.flatten_all(broker, reason="all_unknown")
        assert result.positions_closed == 0
        assert result.positions_failed == 0
        assert broker.placed_orders == []

    def test_fractional_quantity_rounding_to_zero_is_noop(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.seed_position("AAPL", "BUY", 0.4)
        result = ks.flatten_all(broker, reason="micro_position")
        # int(0.4) == 0 → nothing to do, no order placed.
        assert result.positions_closed == 0
        assert result.positions_failed == 0
        assert broker.placed_orders == []

    def test_broker_without_place_equity_order_raises(self, tmp_path):
        ks = _make_kill_switch(tmp_path)

        class BareBroker(PaperEquityBroker):
            place_equity_order = None  # type: ignore[assignment]

        broker = BareBroker()
        broker.seed_position("AAPL", "BUY", 10)
        with pytest.raises(EquityKillSwitchError, match="place_equity_order"):
            ks.flatten_all(broker, reason="bare_broker")


# ---------------------------------------------------------------------------
# flatten_all partial failure
# ---------------------------------------------------------------------------


class TestFlattenAllPartialFailure:
    def test_failed_order_cancellation_raises_partial(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.seed_working_order("11", "AAPL", "BUY", 10)
        broker.seed_working_order("12", "MSFT", "SELL", 20)
        broker.orders_to_fail_close.add("11")

        with pytest.raises(EquityKillSwitchPartialFailure) as exc_info:
            ks.flatten_all(broker, reason="partial_cancel")

        assert exc_info.value.remaining_orders == ["11"]
        assert exc_info.value.remaining_positions == []

    def test_failed_position_flat_raises_partial_with_symbol(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.seed_position("AAPL", "BUY", 10)
        broker.seed_position("MSFT", "BUY", 20)
        broker.positions_to_fail_flat.add("MSFT")

        with pytest.raises(EquityKillSwitchPartialFailure) as exc_info:
            ks.flatten_all(broker, reason="partial_flat")

        assert exc_info.value.remaining_orders == []
        assert exc_info.value.remaining_positions == ["MSFT"]

    def test_get_trades_failure_aborts_with_kill_switch_error(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.get_trades_raises = ConnectionError("broker dead")
        with pytest.raises(EquityKillSwitchError, match="enumerate working orders"):
            ks.flatten_all(broker, reason="dead_broker")

    def test_get_positions_failure_aborts_with_kill_switch_error(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.get_positions_raises = TimeoutError("hung")
        with pytest.raises(EquityKillSwitchError, match="enumerate open positions"):
            ks.flatten_all(broker, reason="hung_broker")

    def test_close_trade_raises_counts_as_failed_not_fatal(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.seed_working_order("99", "AAPL", "BUY", 10)

        # Subclass that raises from close_trade for that specific id.
        class FlakyBroker(PaperEquityBroker):
            def close_trade(self, trade_id):
                self.cancel_calls.append(trade_id)
                raise ConnectionError("flaky")

        flaky = FlakyBroker()
        flaky.seed_working_order("99", "AAPL", "BUY", 10)
        flaky.seed_position("AAPL", "BUY", 10)
        with pytest.raises(EquityKillSwitchPartialFailure) as exc_info:
            ks.flatten_all(flaky, reason="flaky_cancel")
        # The flatten still continued (positions enumerated), and the
        # exception carries the remaining order. Position WAS flattened.
        assert exc_info.value.remaining_orders == ["99"]
        assert exc_info.value.remaining_positions == []


# ---------------------------------------------------------------------------
# Re-poll-to-confirm-flat (M2 fix): "flat" only after the broker confirms the
# book is actually empty across re-polls; never-flat escalates rather than
# reporting success. No mocks — real BrokerClient subclasses with plain flags.
# ---------------------------------------------------------------------------


class DelayedFlatBroker(PaperEquityBroker):
    """Accepts the flatten order but the book clears only after N re-polls.

    Real BrokerClient subclass (no mock). ``place_equity_order`` returns a
    PENDING OrderResult and DOES NOT mutate the position immediately —
    modelling a broker whose flatten order is accepted (PENDING) but whose
    fill confirms asynchronously. ``get_open_positions`` keeps reporting the
    position until it has been polled ``flat_after_polls`` times, then drops
    it (the fill settled). This proves the kill switch credits a close ONLY
    after the broker confirms the symbol is gone from the book.
    """

    def __init__(self, *, flat_after_polls: int) -> None:
        super().__init__()
        self.flat_after_polls = flat_after_polls
        self.position_poll_count = 0
        self._symbols_pending_fill: set = set()

    def place_equity_order(self, instrument, direction, quantity):
        symbol = instrument.symbol
        self.placed_orders.append(
            {
                "symbol": symbol,
                "direction": direction,
                "quantity": quantity,
                "asset_class": instrument.asset_class,
            }
        )
        if symbol in self.positions_to_fail_flat:
            return OrderResult(
                trade_id="0",
                fill_price=0.0,
                status="rejected",
                error_message="paper broker forced rejection",
            )
        # Accepted (PENDING) but NOT yet settled — the position lingers on
        # the book until enough re-polls elapse.
        self._symbols_pending_fill.add(symbol)
        self._next_id += 1
        return OrderResult(
            trade_id=str(self._next_id),
            fill_price=0.0,
            status="PENDING",
        )

    def get_open_positions(self):
        if self.get_positions_raises is not None:
            raise self.get_positions_raises
        self.position_poll_count += 1
        if self.position_poll_count >= self.flat_after_polls:
            # Fill settled — drop every pending-fill symbol from the book.
            for symbol in list(self._symbols_pending_fill):
                self._positions.pop(symbol, None)
            self._symbols_pending_fill.clear()
        return list(self._positions.values())


class NeverFlatBroker(PaperEquityBroker):
    """Accepts the flatten order (PENDING) but the book NEVER clears.

    Real BrokerClient subclass (no mock). Models the worst case for a kill
    switch: the broker takes the order, reports PENDING, yet the position
    stays live indefinitely (rejected-at-exchange, hung fill, partial). The
    kill switch must NOT report success — it must exhaust its re-poll budget
    and escalate.
    """

    def place_equity_order(self, instrument, direction, quantity):
        symbol = instrument.symbol
        self.placed_orders.append(
            {
                "symbol": symbol,
                "direction": direction,
                "quantity": quantity,
                "asset_class": instrument.asset_class,
            }
        )
        self._next_id += 1
        # Accepted, but DELIBERATELY do not touch self._positions — the
        # book stays long forever.
        return OrderResult(
            trade_id=str(self._next_id),
            fill_price=0.0,
            status="PENDING",
        )


class TestConfirmFlatViaRepoll:
    def test_reports_flat_only_after_book_confirmed_empty(self, tmp_path):
        # Order accepted (PENDING) immediately, but the book only clears on
        # the 3rd re-poll. The kill switch must keep polling and credit the
        # close ONLY once the symbol is gone — not on mere acceptance.
        ks = _make_kill_switch(tmp_path)
        broker = DelayedFlatBroker(flat_after_polls=3)
        broker.seed_position("AAPL", "BUY", 40)

        sleeps: List[float] = []
        result = ks.flatten_all(
            broker,
            reason="confirm_after_repoll",
            sleep_fn=lambda d: sleeps.append(d),
            confirm_flat_max_polls=5,
        )

        # Credited closed ONLY after the broker confirmed the book flat.
        assert result.positions_closed == 1
        assert result.positions_failed == 0
        assert result.remaining_position_symbols == []
        # The book is genuinely flat now.
        assert broker.get_open_positions() == []
        # The flatten order was sent exactly ONCE; re-polls do not re-send.
        assert len(broker.placed_orders) == 1
        # It took multiple re-polls to confirm (so we backed off at least
        # once before the book cleared).
        assert broker.position_poll_count >= 3
        assert len(sleeps) >= 1

    def test_acceptance_alone_is_not_credited_as_closed(self, tmp_path):
        # If the book clears only on the LAST allowed poll, the close is
        # still credited — but acceptance on poll 1 must NOT have been
        # enough on its own (the broker kept reporting the position).
        ks = _make_kill_switch(tmp_path)
        broker = DelayedFlatBroker(flat_after_polls=4)
        broker.seed_position("MSFT", "BUY", 10)

        result = ks.flatten_all(
            broker,
            reason="last_poll_flat",
            sleep_fn=lambda _d: None,
            confirm_flat_max_polls=4,
        )
        assert result.positions_closed == 1
        assert result.remaining_position_symbols == []
        assert broker.position_poll_count == 4

    def test_never_flat_escalates_instead_of_reporting_success(self, tmp_path):
        # The broker accepts the order (PENDING) but the position never
        # leaves the book. The kill switch MUST exhaust its re-poll budget
        # and raise — never silently report a flat that isn't real.
        ks = _make_kill_switch(tmp_path)
        broker = NeverFlatBroker()
        broker.seed_position("AAPL", "BUY", 25)

        sleeps: List[float] = []
        with pytest.raises(EquityKillSwitchPartialFailure) as exc_info:
            ks.flatten_all(
                broker,
                reason="never_flat",
                sleep_fn=lambda d: sleeps.append(d),
                confirm_flat_max_polls=3,
            )

        # The un-confirmed symbol is surfaced for the supervisor.
        assert exc_info.value.remaining_positions == ["AAPL"]
        assert exc_info.value.remaining_orders == []
        # Order was accepted (sent once) but exposure remains live.
        assert len(broker.placed_orders) == 1
        assert broker._positions  # still long — proves no false flat
        # Bounded retries: max_polls - 1 backoff sleeps (last poll no sleep).
        assert len(sleeps) == 2

    def test_never_flat_event_log_records_remaining(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = NeverFlatBroker()
        broker.seed_position("AAPL", "BUY", 25)
        with pytest.raises(EquityKillSwitchPartialFailure):
            ks.flatten_all(
                broker,
                reason="never_flat_logged",
                sleep_fn=lambda _d: None,
                confirm_flat_max_polls=2,
            )
        state = json.loads(ks.state_path.read_text(encoding="utf-8"))
        event = state["events"][-1]
        assert event["result"]["positions_closed"] == 0
        assert event["result"]["positions_failed"] == 1
        assert event["result"]["remaining_position_symbols"] == ["AAPL"]

    def test_partial_book_one_flat_one_stuck_escalates(self, tmp_path):
        # Two positions: one flattens cleanly, one never clears. The clean
        # one is credited; the stuck one escalates. A kill switch must not
        # average a partial flat into a green light.
        class MixedBroker(PaperEquityBroker):
            """AAPL flattens (synchronous fill); STUCK never clears."""

            def place_equity_order(self, instrument, direction, quantity):
                symbol = instrument.symbol
                self.placed_orders.append(
                    {
                        "symbol": symbol,
                        "direction": direction,
                        "quantity": quantity,
                        "asset_class": instrument.asset_class,
                    }
                )
                self._next_id += 1
                if symbol == "AAPL":
                    self._positions.pop(symbol, None)
                # STUCK: accepted but left on the book.
                return OrderResult(
                    trade_id=str(self._next_id),
                    fill_price=0.0,
                    status="PENDING",
                )

        ks = _make_kill_switch(tmp_path)
        broker = MixedBroker()
        broker.seed_position("AAPL", "BUY", 10)
        broker.seed_position("STUCK", "BUY", 10)

        with pytest.raises(EquityKillSwitchPartialFailure) as exc_info:
            ks.flatten_all(
                broker,
                reason="mixed_flat",
                sleep_fn=lambda _d: None,
                confirm_flat_max_polls=3,
            )
        assert exc_info.value.remaining_positions == ["STUCK"]
        assert "AAPL" not in exc_info.value.remaining_positions

    def test_rejected_order_not_confirmed_and_never_polled_as_flat(
        self, tmp_path
    ):
        # A rejected flatten order fails up-front and must NOT be credited
        # even though the broker book happens to not contain it.
        ks = _make_kill_switch(tmp_path)
        broker = DelayedFlatBroker(flat_after_polls=1)
        broker.seed_position("AAPL", "BUY", 10)
        broker.positions_to_fail_flat.add("AAPL")
        with pytest.raises(EquityKillSwitchPartialFailure) as exc_info:
            ks.flatten_all(
                broker,
                reason="rejected_flat",
                sleep_fn=lambda _d: None,
            )
        assert exc_info.value.remaining_positions == ["AAPL"]

    def test_repoll_broker_failure_fails_closed(self, tmp_path):
        # If the re-poll itself can't reach the broker, the kill switch
        # must fail closed (raise) rather than assume flat.
        class RepollDiesBroker(NeverFlatBroker):
            def __init__(self):
                super().__init__()
                self._poll_calls = 0

            def get_open_positions(self):
                # First call (the initial enumerate) succeeds; the confirm
                # re-poll then dies.
                self._poll_calls += 1
                if self._poll_calls >= 2:
                    raise ConnectionError("broker dropped during confirm")
                return list(self._positions.values())

        ks = _make_kill_switch(tmp_path)
        broker = RepollDiesBroker()
        broker.seed_position("AAPL", "BUY", 10)
        with pytest.raises(
            EquityKillSwitchError, match="could not confirm flat"
        ):
            ks.flatten_all(
                broker,
                reason="repoll_dies",
                sleep_fn=lambda _d: None,
            )

    def test_invalid_max_polls_refused(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        with pytest.raises(
            EquityKillSwitchError, match="confirm_flat_max_polls"
        ):
            ks.flatten_all(
                PaperEquityBroker(),
                reason="bad_polls",
                confirm_flat_max_polls=0,
            )


# ---------------------------------------------------------------------------
# Persisted event log (atomic writes)
# ---------------------------------------------------------------------------


class TestEventLog:
    def test_event_log_written_after_success(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.seed_position("AAPL", "BUY", 10)
        when = datetime(2026, 6, 21, 16, 30, tzinfo=timezone.utc)
        ks.flatten_all(broker, reason="audit_one", now_fn=lambda: when)

        state = json.loads(ks.state_path.read_text(encoding="utf-8"))
        assert state["version"] == STATE_VERSION
        assert state["universe_hash"] == VALID_HASH
        assert len(state["events"]) == 1
        event = state["events"][0]
        assert event["reason"] == "audit_one"
        assert event["ts"] == when.isoformat()
        assert event["result"]["positions_closed"] == 1

    def test_event_log_written_even_on_partial(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        broker.seed_position("AAPL", "BUY", 10)
        broker.positions_to_fail_flat.add("AAPL")

        with pytest.raises(EquityKillSwitchPartialFailure):
            ks.flatten_all(broker, reason="partial_should_log")
        state = json.loads(ks.state_path.read_text(encoding="utf-8"))
        assert len(state["events"]) == 1
        assert state["events"][0]["reason"] == "partial_should_log"
        assert state["events"][0]["result"]["positions_failed"] == 1
        assert state["events"][0]["result"]["remaining_position_symbols"] == ["AAPL"]

    def test_event_log_appends_across_invocations(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        ks.flatten_all(broker, reason="a")
        ks.flatten_all(broker, reason="b")
        ks.flatten_all(broker, reason="c")
        state = json.loads(ks.state_path.read_text(encoding="utf-8"))
        assert [e["reason"] for e in state["events"]] == ["a", "b", "c"]

    def test_event_log_fifo_cap(self, tmp_path):
        ks = _make_kill_switch(
            tmp_path, max_event_log_entries=2,
        )
        broker = PaperEquityBroker()
        ks.flatten_all(broker, reason="one")
        ks.flatten_all(broker, reason="two")
        ks.flatten_all(broker, reason="three")
        state = json.loads(ks.state_path.read_text(encoding="utf-8"))
        # Oldest dropped; newest two kept.
        assert [e["reason"] for e in state["events"]] == ["two", "three"]

    def test_corrupt_prior_state_rewritten_with_fresh_log(self, tmp_path):
        gate = _ship_gate(tmp_path)
        state_path = tmp_path / "kill_switch_state.json"
        state_path.write_text("{not json", encoding="utf-8")
        ks = EquityKillSwitch(
            universe_hash=VALID_HASH,
            state_path=state_path,
            ship_gate_path=gate,
        )
        broker = PaperEquityBroker()
        ks.flatten_all(broker, reason="recover")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert len(state["events"]) == 1
        assert state["events"][0]["reason"] == "recover"

    def test_mismatched_prior_state_universe_hash_rewritten(self, tmp_path, caplog):
        gate = _ship_gate(tmp_path)
        state_path = tmp_path / "kill_switch_state.json"
        # Pre-existing event log from a DIFFERENT universe must be
        # discarded, not appended to — otherwise the audit trail
        # crosses universes and is meaningless.
        state_path.write_text(
            json.dumps(
                {
                    "version": STATE_VERSION,
                    "universe_hash": "different_hash",
                    "events": [{"ts": "old", "reason": "stale", "result": {}}],
                }
            ),
            encoding="utf-8",
        )
        ks = EquityKillSwitch(
            universe_hash=VALID_HASH,
            state_path=state_path,
            ship_gate_path=gate,
        )
        caplog.set_level(logging.WARNING, logger="src.equity.kill_switch")
        ks.flatten_all(PaperEquityBroker(), reason="fresh")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert len(state["events"]) == 1
        assert state["events"][0]["reason"] == "fresh"

    def test_event_log_atomic_no_partial_file(self, tmp_path):
        ks = _make_kill_switch(tmp_path)
        broker = PaperEquityBroker()
        ks.flatten_all(broker, reason="atomic")
        # After atomic write, no tempfile siblings should remain in the
        # state-file directory.
        leftover = [
            p for p in ks.state_path.parent.iterdir()
            if p.name.startswith(".kill_switch_")
        ]
        assert leftover == [], (
            f"atomic-write tempfile leaked: {leftover}"
        )


# ---------------------------------------------------------------------------
# Source-file hygiene
# ---------------------------------------------------------------------------


def test_kill_switch_source_has_no_bare_except():
    src = Path("src/equity/kill_switch.py").read_text(encoding="utf-8")
    # Match `except:` not followed by an exception name (i.e. bare).
    bare = re.findall(r"^\s*except\s*:\s*$", src, flags=re.MULTILINE)
    assert bare == [], f"bare except in kill_switch.py: {bare}"


def test_kill_switch_source_has_no_unittest_mock_imports():
    src = Path("src/equity/kill_switch.py").read_text(encoding="utf-8")
    forbidden = ("unittest.mock", "MagicMock", "patch(")
    found = [tok for tok in forbidden if tok in src]
    assert found == [], f"forbidden mock tokens in kill_switch.py: {found}"


# NOTE: a self-greping mock-import check for THIS test file is
# intentionally omitted — the file's own module docstring legitimately
# names ``unittest.mock`` to document that we don't use it, which would
# self-trip any plain string search. The implementation-file check
# above covers what matters (no mocks in the production module).


# ---------------------------------------------------------------------------
# Integration: real paper IBGateway (skipped when unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_paper_account_flatten_all_round_trips(tmp_path):
    """End-to-end on a paper IBGateway: buy 1 share -> flatten_all -> flat.

    Skips cleanly when ib_async is not installed or the gateway is not
    reachable. When it runs, validates that:

    * ``broker.arm_equity_gate`` accepts the same ship-gate fixture
      used by every other equity surface.
    * The kill switch's flatten_all submits an opposing market order
      via ``place_equity_order`` (so the audit path is the same one
      the rebalance scheduler uses, not a side door).
    * Post-flatten ``broker.get_open_positions()`` reports no rows for
      the test ticker (or the test ticker's quantity is 0).
    """
    try:
        from src.brokers.ibkr import IBKRBroker, Stock
    except ImportError:
        pytest.skip("src.brokers.ibkr unavailable")
    if Stock is None:
        pytest.skip("ib_async/ib_insync not installed; cannot run paper test")

    host = os.environ.get("IBKR_PAPER_HOST", "127.0.0.1")
    port = int(os.environ.get("IBKR_PAPER_PORT", "7497"))
    client_id = int(os.environ.get("IBKR_PAPER_CLIENT_ID", "201"))
    symbol = os.environ.get("IBKR_PAPER_SYMBOL", "AAPL")

    gate = _ship_gate(tmp_path)
    broker = IBKRBroker(host=host, port=port, client_id=client_id)
    broker.arm_equity_gate(VALID_HASH, ship_gate_path=gate)

    try:
        broker.connect()
    except (ConnectionError, OSError) as exc:
        pytest.skip(f"IBGateway not reachable at {host}:{port} ({exc})")

    placed_status: Optional[str] = None
    try:
        inst = Instrument.equity(symbol)
        place_result = broker.place_equity_order(inst, "BUY", 1)
        placed_status = place_result.status
        if placed_status != "PENDING":
            pytest.skip(
                f"paper broker rejected setup order: {place_result.error_message!r}"
            )

        ks = EquityKillSwitch(
            universe_hash=VALID_HASH,
            state_path=tmp_path / "ks.json",
            ship_gate_path=gate,
        )
        try:
            ks.flatten_all(broker, reason="integration_paper")
        except EquityKillSwitchPartialFailure as partial:
            # The paper account may settle the BUY before the SELL
            # submits; partial here means at least one flat WAS sent.
            assert partial.remaining_positions or partial.remaining_orders
        # Verify the ticker is no longer net-long. We accept either
        # 0 rows for the symbol OR quantity == 0 (paper accounts can
        # leave a 0-quantity row briefly).
        positions = {p.instrument: p for p in broker.get_open_positions()}
        if symbol in positions:
            assert positions[symbol].quantity == 0
    finally:
        try:
            broker.disconnect()
        except (ConnectionError, OSError):
            pass
