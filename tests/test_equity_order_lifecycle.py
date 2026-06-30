"""US-012 — IBKR equity order lifecycle tests (no mocks, real disk).

Drives :mod:`src.equity.order_lifecycle` end-to-end against ``tmp_path``
real-disk fixtures. Asserts:

* Policy types/TIF defaults match the documented matrix
* Whole-share rounding floors quantity and surfaces residual cash
* SUBMITTED/PARTIAL/FILLED/FAILED transitions are validated
* Forced reconcile mismatch -> FAILED with reason_code, logged at ERROR
* ``place_with_retry`` retries on connection drop and raises the named
  :class:`BrokerTimeoutError` after ``MAX_RETRIES`` attempts
* Ship-gate guard rejects missing/false/mismatched gate files
* Ledger persistence is atomic and survives restart
* No bare ``except:`` clauses exist in ``src/brokers/ibkr.py`` or
  ``src/equity/order_lifecycle.py`` (greps the on-disk source files)

No mocks, no ``unittest.mock``/``MagicMock``/``patch``. The retry test
uses a tiny stateful callable plus an in-process sleep counter — both
real objects, no test doubles.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

import pytest

from src.brokers.types import OrderResult
from src.equity.order_lifecycle import (
    MAX_RETRIES,
    SHIP_GATE_PATH_DEFAULT,
    STATE_VERSION,
    BrokerTimeoutError,
    EquityOrder,
    OrderLifecycle,
    OrderLifecycleError,
    OrderState,
    OrderType,
    TIF,
    WholeShareDecision,
    enforce_ship_gate,
    whole_share_round,
)


VALID_HASH = "8bfe419a0c1c06306ff3fa35c39132df522b17ae31718a1dd5463097b467899c"


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Mirror the production atomic-write pattern for the fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".test_lifecycle_", suffix=".json", dir=str(path.parent),
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


def _ship_gate(tmp_path: Path, *, hash_: str = VALID_HASH) -> Path:
    path = tmp_path / "SHIP_GATE.json"
    _atomic_write_json(path, _passing_payload(universe_hash=hash_))
    return path


def _make_lifecycle(tmp_path: Path, **kwargs) -> OrderLifecycle:
    gate = _ship_gate(tmp_path)
    return OrderLifecycle(
        universe_hash=VALID_HASH,
        state_path=tmp_path / "lifecycle.json",
        ship_gate_path=gate,
        retry_base_delay_s=0.0,  # tests run fast
        retry_max_delay_s=0.0,
        sleep_fn=lambda _: None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Documented policy + constants
# ---------------------------------------------------------------------------


class TestDocumentedPolicy:
    """AC #1 — documented policy: types, TIF, whole-share rule."""

    def test_order_types_enumerated(self):
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert set(OrderType) == {OrderType.MARKET, OrderType.LIMIT}

    def test_tif_enumerated(self):
        assert TIF.DAY.value == "DAY"
        assert TIF.GTC.value == "GTC"
        assert set(TIF) == {TIF.DAY, TIF.GTC}

    def test_states_enumerated(self):
        assert {s.value for s in OrderState} == {
            "SUBMITTED", "PARTIAL", "FILLED", "FAILED",
        }

    def test_default_order_is_market_day(self):
        order = EquityOrder(
            client_order_id="cid-1",
            ticker="AAPL",
            side="BUY",
            quantity=10,
        )
        assert order.order_type == OrderType.MARKET.value
        assert order.time_in_force == TIF.DAY.value
        assert order.state == OrderState.SUBMITTED.value

    def test_docstring_documents_policy(self):
        # AC #1 mandates the policy be documented (NOT the FX bracket).
        # The docstring is the documentation — assert key strings live
        # in it so a future edit can't silently drop them.
        from src.equity import order_lifecycle as ol_mod
        doc = ol_mod.__doc__ or ""
        assert "MARKET" in doc and "LIMIT" in doc
        assert "DAY" in doc and "GTC" in doc
        assert "whole" in doc.lower() and "residual" in doc.lower()
        # Distinguished from the FX bracket — assert that contrast is
        # spelled out.
        assert "bracket" in doc.lower()


# ---------------------------------------------------------------------------
# Whole-share rounding + residual cash
# ---------------------------------------------------------------------------


class TestWholeShareRounding:
    """AC #1 — whole-share rounding with a residual-cash rule."""

    def test_rounds_down_to_whole_shares(self):
        d = whole_share_round("AAPL", target_weight=0.10, nav=10_000, price=33.33)
        # target_cash = 1000; 1000 / 33.33 = 30.0030 -> floor -> 30
        assert d.shares == 30
        # residual = 1000 - 30*33.33 = 1000 - 999.9 = 0.1
        assert d.residual_cash == pytest.approx(0.1, abs=1e-9)

    def test_zero_weight_yields_zero_shares_zero_residual(self):
        d = whole_share_round("AAPL", target_weight=0.0, nav=10_000, price=100)
        assert d.shares == 0
        assert d.residual_cash == 0.0

    def test_negative_weight_is_clamped_long_only(self):
        d = whole_share_round("AAPL", target_weight=-0.5, nav=10_000, price=50)
        assert d.shares == 0
        assert d.residual_cash == 0.0
        assert d.target_weight == 0.0

    def test_exact_fit_has_zero_residual(self):
        d = whole_share_round("AAPL", target_weight=0.5, nav=1000, price=50)
        # 0.5 * 1000 / 50 = 10 shares exactly, no residual.
        assert d.shares == 10
        assert d.residual_cash == pytest.approx(0.0, abs=1e-9)

    def test_zero_nav_refuses(self):
        with pytest.raises(OrderLifecycleError, match="NAV must be > 0"):
            whole_share_round("AAPL", target_weight=0.1, nav=0, price=100)

    def test_negative_nav_refuses(self):
        with pytest.raises(OrderLifecycleError, match="NAV must be > 0"):
            whole_share_round("AAPL", target_weight=0.1, nav=-1, price=100)

    def test_zero_price_refuses(self):
        with pytest.raises(OrderLifecycleError, match="price must be > 0"):
            whole_share_round("AAPL", target_weight=0.1, nav=1000, price=0)

    def test_negative_price_refuses(self):
        with pytest.raises(OrderLifecycleError, match="price must be > 0"):
            whole_share_round("AAPL", target_weight=0.1, nav=1000, price=-50)

    def test_residual_summary_aggregates(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        decisions = [
            whole_share_round("AAPL", 0.10, 10_000, 33.33),
            whole_share_round("MSFT", 0.10, 10_000, 250.0),  # 4 shares, res 0
        ]
        summary = lc.residual_cash_summary(decisions)
        assert "total" in summary and "per_ticker" in summary
        assert summary["per_ticker"]["AAPL"] == pytest.approx(0.1, abs=1e-9)
        assert summary["per_ticker"]["MSFT"] == pytest.approx(0.0, abs=1e-9)
        assert summary["total"] == pytest.approx(
            summary["per_ticker"]["AAPL"] + summary["per_ticker"]["MSFT"],
            abs=1e-9,
        )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    """AC #2 — SUBMITTED/PARTIAL/FILLED/FAILED transitions validated."""

    def _register(self, lc: OrderLifecycle, **kwargs) -> EquityOrder:
        order = EquityOrder(
            client_order_id=kwargs.pop("client_order_id", "cid-1"),
            ticker=kwargs.pop("ticker", "AAPL"),
            side=kwargs.pop("side", "BUY"),
            quantity=kwargs.pop("quantity", 10),
            **kwargs,
        )
        return lc.register(order)

    def test_happy_path_submitted_to_filled(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        result = lc.transition(
            "cid-1", OrderState.FILLED, filled_quantity=10,
        )
        assert result.state == OrderState.FILLED.value
        assert result.filled_quantity == 10

    def test_partial_then_filled(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        lc.transition("cid-1", OrderState.PARTIAL, filled_quantity=4)
        assert lc.get("cid-1").state == OrderState.PARTIAL.value
        assert lc.get("cid-1").filled_quantity == 4
        lc.transition("cid-1", OrderState.FILLED, filled_quantity=10)
        assert lc.get("cid-1").state == OrderState.FILLED.value

    def test_partial_requires_positive_qty(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        with pytest.raises(OrderLifecycleError, match="PARTIAL.*filled_quantity"):
            lc.transition("cid-1", OrderState.PARTIAL, filled_quantity=0)

    def test_partial_requires_strictly_less_than_quantity(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        with pytest.raises(OrderLifecycleError, match="PARTIAL"):
            lc.transition("cid-1", OrderState.PARTIAL, filled_quantity=10)

    def test_filled_requires_full_quantity(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        with pytest.raises(OrderLifecycleError, match="FILLED"):
            lc.transition("cid-1", OrderState.FILLED, filled_quantity=5)

    def test_failed_requires_reason_code(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        with pytest.raises(OrderLifecycleError, match="reason_code"):
            lc.transition("cid-1", OrderState.FAILED)

    def test_failed_logs_error_with_reason_code(self, tmp_path, caplog):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        with caplog.at_level(logging.ERROR, logger="src.equity.order_lifecycle"):
            lc.transition(
                "cid-1", OrderState.FAILED, reason_code="ibkr_201",
            )
        assert any(
            "FAILED" in rec.getMessage() and "ibkr_201" in rec.getMessage()
            and rec.levelno >= logging.ERROR
            for rec in caplog.records
        )
        assert lc.get("cid-1").reason_code == "ibkr_201"

    def test_filled_terminal_cannot_re_transition(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        lc.transition("cid-1", OrderState.FILLED, filled_quantity=10)
        with pytest.raises(OrderLifecycleError, match="illegal transition"):
            lc.transition(
                "cid-1", OrderState.FAILED, reason_code="too_late",
            )

    def test_failed_terminal_cannot_re_transition(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        lc.transition("cid-1", OrderState.FAILED, reason_code="early_reject")
        with pytest.raises(OrderLifecycleError, match="illegal transition"):
            lc.transition("cid-1", OrderState.FILLED, filled_quantity=10)

    def test_submitted_cannot_skip_back_to_submitted(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        with pytest.raises(OrderLifecycleError, match="illegal transition"):
            lc.transition("cid-1", OrderState.SUBMITTED)

    def test_register_validates_side(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        with pytest.raises(OrderLifecycleError, match="side"):
            lc.register(EquityOrder(
                client_order_id="cid-1",
                ticker="AAPL",
                side="LONG",
                quantity=10,
            ))

    def test_register_validates_quantity(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        with pytest.raises(OrderLifecycleError, match="quantity"):
            lc.register(EquityOrder(
                client_order_id="cid-1",
                ticker="AAPL",
                side="BUY",
                quantity=0,
            ))

    def test_register_validates_limit_price(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        with pytest.raises(OrderLifecycleError, match="LIMIT"):
            lc.register(EquityOrder(
                client_order_id="cid-1",
                ticker="AAPL",
                side="BUY",
                quantity=10,
                order_type=OrderType.LIMIT.value,
                limit_price=None,
            ))

    def test_register_validates_tif(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        with pytest.raises(OrderLifecycleError, match="time_in_force"):
            lc.register(EquityOrder(
                client_order_id="cid-1",
                ticker="AAPL",
                side="BUY",
                quantity=10,
                time_in_force="IOC",
            ))

    def test_register_idempotent_on_known_id(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        first = self._register(lc)
        again = self._register(lc, ticker="MSFT")  # different ticker
        # Idempotent: same client_order_id returns the SAME registered
        # record; the second call's payload is intentionally ignored.
        assert again is first
        assert lc.get("cid-1").ticker == "AAPL"


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


class TestReconciliation:
    """AC #2 — partial-fill handling + forced-mismatch -> FAILED."""

    def _register(self, lc: OrderLifecycle) -> EquityOrder:
        return lc.register(EquityOrder(
            client_order_id="cid-1",
            ticker="AAPL",
            side="BUY",
            quantity=10,
        ))

    def test_no_fill_yet_keeps_submitted(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        order = lc.reconcile("cid-1", broker_filled_quantity=0)
        assert order.state == OrderState.SUBMITTED.value
        assert order.filled_quantity == 0

    def test_partial_fill_transitions_to_partial(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        order = lc.reconcile("cid-1", broker_filled_quantity=4)
        assert order.state == OrderState.PARTIAL.value
        assert order.filled_quantity == 4

    def test_full_fill_transitions_to_filled(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        order = lc.reconcile("cid-1", broker_filled_quantity=10)
        assert order.state == OrderState.FILLED.value
        assert order.filled_quantity == 10

    def test_partial_then_full_progresses(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        lc.reconcile("cid-1", broker_filled_quantity=4)
        order = lc.reconcile("cid-1", broker_filled_quantity=10)
        assert order.state == OrderState.FILLED.value

    def test_mismatch_overshoot_forces_failed(self, tmp_path, caplog):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        with caplog.at_level(logging.ERROR, logger="src.equity.order_lifecycle"):
            order = lc.reconcile("cid-1", broker_filled_quantity=11)
        assert order.state == OrderState.FAILED.value
        assert order.reason_code == "reconcile_mismatch"
        assert any(
            "reconcile mismatch" in rec.getMessage() and rec.levelno >= logging.ERROR
            for rec in caplog.records
        )

    def test_mismatch_regression_forces_failed(self, tmp_path, caplog):
        # Ledger says we already filled 6; broker now reports only 3.
        # That's a hard inconsistency; never recover silently.
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        lc.reconcile("cid-1", broker_filled_quantity=6)
        assert lc.get("cid-1").state == OrderState.PARTIAL.value
        with caplog.at_level(logging.ERROR, logger="src.equity.order_lifecycle"):
            order = lc.reconcile("cid-1", broker_filled_quantity=3)
        assert order.state == OrderState.FAILED.value
        assert order.reason_code == "reconcile_mismatch"

    def test_negative_broker_qty_forces_failed(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        order = lc.reconcile("cid-1", broker_filled_quantity=-1)
        assert order.state == OrderState.FAILED.value
        assert order.reason_code == "reconcile_negative_qty"

    def test_filled_terminal_reconciles_idempotently_on_match(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        lc.transition("cid-1", OrderState.FILLED, filled_quantity=10)
        order = lc.reconcile("cid-1", broker_filled_quantity=10)
        assert order.state == OrderState.FILLED.value  # no-op

    def test_filled_terminal_with_mismatch_forces_failed(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        self._register(lc)
        lc.transition("cid-1", OrderState.FILLED, filled_quantity=10)
        order = lc.reconcile("cid-1", broker_filled_quantity=7)
        assert order.state == OrderState.FAILED.value
        assert order.reason_code == "reconcile_mismatch"


# ---------------------------------------------------------------------------
# Retry + BrokerTimeoutError
# ---------------------------------------------------------------------------


class _FlakySender:
    """Fail the first ``n`` calls with ``exc_cls``, then return ``result``."""

    def __init__(self, fail_count: int, exc_cls: type, result: OrderResult):
        self.fail_count = fail_count
        self.exc_cls = exc_cls
        self.result = result
        self.call_count = 0

    def __call__(self) -> OrderResult:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise self.exc_cls("simulated transient drop")
        return self.result


class _AlwaysFails:
    """Always fails with the given exception class — used for retry-exhaust."""

    def __init__(self, exc_cls: type):
        self.exc_cls = exc_cls
        self.call_count = 0

    def __call__(self) -> OrderResult:
        self.call_count += 1
        raise self.exc_cls("simulated permanent drop")


class _SleepRecorder:
    """Real callable that records every sleep request (zero wall time)."""

    def __init__(self) -> None:
        self.sleeps: list = []

    def __call__(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))


class TestRetryEnvelope:
    """AC #3 — connection-drop -> retries up to MAX_RETRIES, raises BrokerTimeoutError."""

    def _lc(self, tmp_path: Path) -> OrderLifecycle:
        return _make_lifecycle(tmp_path)

    def test_succeeds_on_first_attempt(self, tmp_path):
        lc = self._lc(tmp_path)
        good_result = OrderResult(
            trade_id="42", fill_price=0.0, status="PENDING",
        )
        sender = _FlakySender(0, ConnectionError, good_result)
        out = lc.place_with_retry(sender)
        assert out is good_result
        assert sender.call_count == 1

    def test_retries_then_succeeds(self, tmp_path):
        sleeps = _SleepRecorder()
        gate = _ship_gate(tmp_path)
        lc = OrderLifecycle(
            universe_hash=VALID_HASH,
            state_path=tmp_path / "lifecycle.json",
            ship_gate_path=gate,
            retry_base_delay_s=0.05,
            retry_max_delay_s=0.05,
            sleep_fn=sleeps,
        )
        good_result = OrderResult(
            trade_id="43", fill_price=0.0, status="PENDING",
        )
        sender = _FlakySender(MAX_RETRIES - 1, ConnectionError, good_result)
        out = lc.place_with_retry(sender)
        assert out is good_result
        assert sender.call_count == MAX_RETRIES
        # Should have slept MAX_RETRIES - 1 times between attempts.
        assert len(sleeps.sleeps) == MAX_RETRIES - 1

    def test_exhausts_retries_and_raises_broker_timeout_error(self, tmp_path):
        sleeps = _SleepRecorder()
        gate = _ship_gate(tmp_path)
        lc = OrderLifecycle(
            universe_hash=VALID_HASH,
            state_path=tmp_path / "lifecycle.json",
            ship_gate_path=gate,
            retry_base_delay_s=0.0,
            retry_max_delay_s=0.0,
            sleep_fn=sleeps,
        )
        sender = _AlwaysFails(ConnectionError)
        with pytest.raises(BrokerTimeoutError) as excinfo:
            lc.place_with_retry(sender)
        # Test asserts retry count AND exception type (AC #3).
        assert isinstance(excinfo.value, BrokerTimeoutError)
        assert excinfo.value.attempts == MAX_RETRIES
        assert sender.call_count == MAX_RETRIES
        # Chained from the original ConnectionError so root-cause grep works.
        assert isinstance(excinfo.value.__cause__, ConnectionError)

    def test_timeout_error_also_triggers_retry(self, tmp_path):
        # Connection-drop AC says "simulated connection drop"; we use
        # TimeoutError too because IBKR raises both for the same root
        # cause (TCP RST vs socket timeout).
        gate = _ship_gate(tmp_path)
        lc = OrderLifecycle(
            universe_hash=VALID_HASH,
            state_path=tmp_path / "lifecycle.json",
            ship_gate_path=gate,
            retry_base_delay_s=0.0,
            retry_max_delay_s=0.0,
            sleep_fn=lambda _: None,
        )
        sender = _AlwaysFails(TimeoutError)
        with pytest.raises(BrokerTimeoutError):
            lc.place_with_retry(sender)
        assert sender.call_count == MAX_RETRIES

    def test_non_transient_exception_surfaces_immediately(self, tmp_path):
        # ValueError is NOT a connectivity error — must not be retried;
        # masking it behind retry noise would hide bugs.
        lc = self._lc(tmp_path)

        class _ValueErrSender:
            def __init__(self):
                self.call_count = 0

            def __call__(self):
                self.call_count += 1
                raise ValueError("bad input")

        sender = _ValueErrSender()
        with pytest.raises(ValueError, match="bad input"):
            lc.place_with_retry(sender)
        assert sender.call_count == 1  # no retry

    def test_max_retries_constant_at_least_two(self):
        # The AC says "retries up to MAX_RETRIES" — assert the constant
        # is set to a value that actually exercises retry behavior.
        assert MAX_RETRIES >= 2


# ---------------------------------------------------------------------------
# Ship-gate guard semantics
# ---------------------------------------------------------------------------


class TestShipGateGuard:
    """AC #4 — module refuses unless SHIP_GATE.json + gate_pass + hash match."""

    def test_passes_when_gate_valid(self, tmp_path):
        gate = _ship_gate(tmp_path)
        enforce_ship_gate(gate, VALID_HASH)  # no raise = pass

    def test_blocks_when_missing(self, tmp_path):
        with pytest.raises(OrderLifecycleError, match="file not found"):
            enforce_ship_gate(tmp_path / "missing.json", VALID_HASH)

    def test_blocks_when_malformed(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(OrderLifecycleError, match="cannot read"):
            enforce_ship_gate(path, VALID_HASH)

    def test_blocks_when_payload_is_list(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        path.write_text("[1,2,3]", encoding="utf-8")
        with pytest.raises(OrderLifecycleError, match="not a JSON object"):
            enforce_ship_gate(path, VALID_HASH)

    def test_blocks_when_gate_pass_false(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        payload = _passing_payload()
        payload["gate_pass"] = False
        _atomic_write_json(path, payload)
        with pytest.raises(OrderLifecycleError, match="gate_pass="):
            enforce_ship_gate(path, VALID_HASH)

    def test_blocks_when_gate_pass_truthy_but_not_true(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        payload = _passing_payload()
        payload["gate_pass"] = 1  # truthy but not identity-True
        _atomic_write_json(path, payload)
        with pytest.raises(OrderLifecycleError, match="gate_pass="):
            enforce_ship_gate(path, VALID_HASH)

    def test_blocks_when_universe_hash_mismatches(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, _passing_payload(universe_hash="ffff"))
        with pytest.raises(OrderLifecycleError, match="universe_hash mismatch"):
            enforce_ship_gate(path, VALID_HASH)

    def test_blocks_when_universe_hash_blank(self, tmp_path):
        path = tmp_path / "SHIP_GATE.json"
        _atomic_write_json(path, _passing_payload(universe_hash=""))
        with pytest.raises(OrderLifecycleError, match="universe_hash missing"):
            enforce_ship_gate(path, VALID_HASH)

    def test_lifecycle_construction_blocks_when_gate_missing(self, tmp_path):
        with pytest.raises(OrderLifecycleError, match="file not found"):
            OrderLifecycle(
                universe_hash=VALID_HASH,
                state_path=tmp_path / "lifecycle.json",
                ship_gate_path=tmp_path / "no_such.json",
                sleep_fn=lambda _: None,
            )

    def test_lifecycle_construction_blocks_when_gate_pass_false(self, tmp_path):
        gate = tmp_path / "SHIP_GATE.json"
        payload = _passing_payload()
        payload["gate_pass"] = False
        _atomic_write_json(gate, payload)
        with pytest.raises(OrderLifecycleError, match="gate_pass="):
            OrderLifecycle(
                universe_hash=VALID_HASH,
                state_path=tmp_path / "lifecycle.json",
                ship_gate_path=gate,
                sleep_fn=lambda _: None,
            )

    def test_lifecycle_construction_blocks_on_hash_mismatch(self, tmp_path):
        gate = _ship_gate(tmp_path, hash_="ffff")
        with pytest.raises(OrderLifecycleError, match="universe_hash mismatch"):
            OrderLifecycle(
                universe_hash=VALID_HASH,
                state_path=tmp_path / "lifecycle.json",
                ship_gate_path=gate,
                sleep_fn=lambda _: None,
            )

    def test_default_path_constant_points_to_canonical_location(self):
        assert SHIP_GATE_PATH_DEFAULT.endswith("SHIP_GATE.json")
        assert "trained_data/backtests" in SHIP_GATE_PATH_DEFAULT


# ---------------------------------------------------------------------------
# Persistence + restart
# ---------------------------------------------------------------------------


class TestPersistenceAndRestart:
    """Atomic state writes + idempotent restart."""

    def test_persistence_round_trip(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        lc.register(EquityOrder(
            client_order_id="cid-1",
            ticker="AAPL",
            side="BUY",
            quantity=10,
        ))
        lc.transition("cid-1", OrderState.PARTIAL, filled_quantity=3)

        # Rehydrate a brand-new instance from the same on-disk state.
        gate = tmp_path / "SHIP_GATE.json"
        lc2 = OrderLifecycle(
            universe_hash=VALID_HASH,
            state_path=tmp_path / "lifecycle.json",
            ship_gate_path=gate,
            sleep_fn=lambda _: None,
        )
        order = lc2.get("cid-1")
        assert order.state == OrderState.PARTIAL.value
        assert order.filled_quantity == 3

    def test_persisted_file_has_version_and_hash(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        lc.register(EquityOrder(
            client_order_id="cid-1",
            ticker="AAPL",
            side="BUY",
            quantity=10,
        ))
        payload = json.loads(
            (tmp_path / "lifecycle.json").read_text(encoding="utf-8")
        )
        assert payload["version"] == STATE_VERSION
        assert payload["universe_hash"] == VALID_HASH
        assert isinstance(payload["orders"], list)
        assert len(payload["orders"]) == 1

    def test_persisted_file_is_sorted_and_indented(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        lc.register(EquityOrder(
            client_order_id="cid-1",
            ticker="AAPL",
            side="BUY",
            quantity=10,
        ))
        text = (tmp_path / "lifecycle.json").read_text(encoding="utf-8")
        # Indented (humans grep it) + keys sorted (no spurious diffs).
        assert "\n  " in text
        # sort_keys=True puts top-level keys alphabetically: orders,
        # universe_hash, version.
        idx_orders = text.find('"orders"')
        idx_uh = text.find('"universe_hash"')
        idx_ver = text.find('"version"')
        assert 0 <= idx_orders < idx_uh < idx_ver

    def test_restart_refuses_universe_hash_change(self, tmp_path):
        lc = _make_lifecycle(tmp_path)
        lc.register(EquityOrder(
            client_order_id="cid-1",
            ticker="AAPL",
            side="BUY",
            quantity=10,
        ))
        # New gate with a DIFFERENT hash — restart must refuse.
        new_gate = tmp_path / "SHIP_GATE2.json"
        _atomic_write_json(new_gate, _passing_payload(universe_hash="abcd"))
        with pytest.raises(OrderLifecycleError, match="universe_hash mismatch"):
            OrderLifecycle(
                universe_hash="abcd",
                state_path=tmp_path / "lifecycle.json",
                ship_gate_path=new_gate,
                sleep_fn=lambda _: None,
            )

    def test_corrupted_state_file_raises(self, tmp_path):
        gate = _ship_gate(tmp_path)
        (tmp_path / "lifecycle.json").write_text("garbage", encoding="utf-8")
        with pytest.raises(OrderLifecycleError, match="not valid JSON"):
            OrderLifecycle(
                universe_hash=VALID_HASH,
                state_path=tmp_path / "lifecycle.json",
                ship_gate_path=gate,
                sleep_fn=lambda _: None,
            )


# ---------------------------------------------------------------------------
# No-bare-except grep (AC #3 grep confirms zero bare except in ibkr.py)
# ---------------------------------------------------------------------------


class TestNoBareExcept:
    """Source-on-disk grep — no test doubles, just read the file."""

    BARE_EXCEPT_RE = re.compile(r"^\s*except\s*:\s*(#.*)?$", re.MULTILINE)

    def _root(self) -> Path:
        # tests/ -> project root
        return Path(__file__).resolve().parent.parent

    def test_no_bare_except_in_ibkr(self):
        ibkr = self._root() / "src" / "brokers" / "ibkr.py"
        text = ibkr.read_text(encoding="utf-8")
        hits = self.BARE_EXCEPT_RE.findall(text)
        assert not hits, (
            f"src/brokers/ibkr.py contains bare except: clauses "
            f"({len(hits)} found)"
        )

    def test_no_bare_except_in_order_lifecycle(self):
        # Strip the tests-internal helper that catches OSError under
        # tempfile cleanup; the lifecycle module itself must be clean.
        ol = self._root() / "src" / "equity" / "order_lifecycle.py"
        text = ol.read_text(encoding="utf-8")
        hits = self.BARE_EXCEPT_RE.findall(text)
        assert not hits, (
            f"src/equity/order_lifecycle.py contains bare except: clauses "
            f"({len(hits)} found)"
        )

    def test_no_unittest_mock_in_order_lifecycle_source(self):
        # The no-mock rule is project-wide; assert the lifecycle
        # implementation itself doesn't import unittest.mock.
        ol = self._root() / "src" / "equity" / "order_lifecycle.py"
        text = ol.read_text(encoding="utf-8")
        forbidden = [
            "import unittest.mock",
            "from unittest import mock",
            "from unittest.mock",
            "MagicMock",
        ]
        for needle in forbidden:
            assert needle not in text, (
                f"{ol} contains forbidden mock reference {needle!r}"
            )
