from __future__ import annotations

from pathlib import Path

from nautilus_trader.core import UUID4
from nautilus_trader.model import AccountId
from nautilus_trader.model import ClientOrderId
from nautilus_trader.model import Currency
from nautilus_trader.model import InstrumentId
from nautilus_trader.model import LiquiditySide
from nautilus_trader.model import Money
from nautilus_trader.model import OrderAccepted
from nautilus_trader.model import OrderCanceled
from nautilus_trader.model import OrderCancelRejected
from nautilus_trader.model import OrderFilled
from nautilus_trader.model import OrderPendingCancel
from nautilus_trader.model import OrderSide
from nautilus_trader.model import OrderSubmitted
from nautilus_trader.model import OrderType
from nautilus_trader.model import Price
from nautilus_trader.model import Quantity
from nautilus_trader.model import TradeId
from nautilus_trader.model import VenueOrderId

from axiom2.execution.reconciliation import Account
from axiom2.execution.reconciliation import Fill
from axiom2.execution.reconciliation import Ledger
from axiom2.execution.reconciliation import Observations
from axiom2.execution.reconciliation import Policy
from axiom2.execution.reconciliation import reconcile
from axiom2.execution.lifecycle import TRANSITIONS
from axiom2.nautilus_runtime import NautilusOrderReplay


def event_uuid(index: int) -> UUID4:
    return UUID4.from_str(f"00000000-0000-4000-8000-{index:012d}")


TRADER_ID = "TRADER-AXIOM2-REPLAY"
STRATEGY_ID = "S-AXIOM2-REPLAY"
INSTRUMENT_ID = InstrumentId.from_str("AAPL.SIM")
ACCOUNT_ID = AccountId("SIM-AXIOM2")
CLIENT_ORDER_ID = ClientOrderId("O-AXIOM2-REPLAY")
VENUE_ORDER_ID = VenueOrderId("VENUE-AXIOM2-1")


def common(index: int) -> dict[str, object]:
    return {
        "trader_id": TRADER_ID,
        "strategy_id": STRATEGY_ID,
        "instrument_id": INSTRUMENT_ID,
        "client_order_id": CLIENT_ORDER_ID,
        "event_id": event_uuid(index),
        "ts_event": index,
        "ts_init": index,
    }


def submitted(index: int) -> OrderSubmitted:
    return OrderSubmitted(**common(index), account_id=ACCOUNT_ID)


def accepted(index: int) -> OrderAccepted:
    return OrderAccepted(
        **common(index),
        venue_order_id=VENUE_ORDER_ID,
        account_id=ACCOUNT_ID,
        reconciliation=False,
    )


def filled(
    index: int,
    trade_id: str,
    quantity: int,
    *,
    price: str = "10.00",
) -> OrderFilled:
    return OrderFilled(
        **common(index),
        venue_order_id=VENUE_ORDER_ID,
        account_id=ACCOUNT_ID,
        trade_id=TradeId(trade_id),
        order_side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        last_qty=Quantity.from_int(quantity),
        last_px=Price.from_str(price),
        currency=Currency.from_str("USD"),
        liquidity_side=LiquiditySide.MAKER,
        reconciliation=False,
        commission=Money.from_str("0.10 USD"),
    )


def pending_cancel(index: int) -> OrderPendingCancel:
    return OrderPendingCancel(
        **common(index),
        account_id=ACCOUNT_ID,
        venue_order_id=VENUE_ORDER_ID,
        reconciliation=False,
    )


def cancel_rejected(index: int) -> OrderCancelRejected:
    return OrderCancelRejected(
        **common(index),
        reason="CANCEL_REJECTED_FOR_REPLAY",
        reconciliation=False,
        venue_order_id=VENUE_ORDER_ID,
        account_id=ACCOUNT_ID,
    )


def canceled(index: int) -> OrderCanceled:
    return OrderCanceled(
        **common(index),
        reconciliation=False,
        venue_order_id=VENUE_ORDER_ID,
        account_id=ACCOUNT_ID,
        reason="REPLAY_CANCEL",
    )


def fill_contract(fill_event: OrderFilled, event_id: str, quantity: int) -> Fill:
    return Fill(
        event_id=event_id,
        order_id=CLIENT_ORDER_ID.value,
        instrument=INSTRUMENT_ID.value,
        quantity=quantity,
        price_cents=int(fill_event.last_px.as_decimal() * 100),
        fee_cents=int(fill_event.commission.raw),
    )


def test_actual_nautilus_order_lifecycle_and_restart_recovery(tmp_path: Path) -> None:
    database = tmp_path / "order-replay.sqlite"
    replay = NautilusOrderReplay(database)

    assert replay.ingest("submitted", submitted(1), received_at_ns=1).snapshot.status == "SUBMITTED"
    assert replay.ingest("accepted", accepted(2), received_at_ns=2).snapshot.status == "ACCEPTED"

    partial = filled(3, "TRADE-PARTIAL", 40)
    assert replay.ingest("fill-partial", partial, received_at_ns=3).snapshot.status == "PARTIALLY_FILLED"
    assert replay.snapshot().filled_quantity_raw == 40

    conflict = filled(4, "TRADE-PARTIAL", 40, price="10.01")
    conflict_result = replay.ingest("fill-partial", conflict, received_at_ns=4)
    assert conflict_result.disposition == "CONFLICT"
    assert replay.snapshot().filled_quantity_raw == 40

    engine_duplicate = filled(5, "TRADE-PARTIAL", 40)
    duplicate_result = replay.ingest(
        "fill-duplicate-engine", engine_duplicate, received_at_ns=5
    )
    assert duplicate_result.disposition == "REJECTED"
    assert duplicate_result.reason is not None
    assert "Duplicate fill" in duplicate_result.reason
    assert replay.snapshot().filled_quantity_raw == 40

    assert replay.ingest("pending-cancel-1", pending_cancel(6), received_at_ns=6).snapshot.status == "PENDING_CANCEL"
    assert replay.ingest("cancel-rejected", cancel_rejected(7), received_at_ns=7).snapshot.status == "PARTIALLY_FILLED"
    assert replay.ingest("pending-cancel-2", pending_cancel(8), received_at_ns=8).snapshot.status == "PENDING_CANCEL"
    assert replay.ingest("canceled", canceled(9), received_at_ns=9).snapshot.status == "CANCELED"

    late = filled(10, "TRADE-LATE", 60)
    assert replay.ingest("fill-late", late, received_at_ns=10).snapshot.status == "FILLED"
    assert replay.snapshot().filled_quantity_raw == 100
    assert replay.snapshot().is_closed
    assert replay.snapshot().is_canceled is False

    duplicate_report = replay.ingest("fill-late", late, received_at_ns=11)
    assert duplicate_report.disposition == "DUPLICATE"
    assert replay.snapshot().filled_quantity_raw == 100

    raw = replay.raw_observations()
    assert [row["disposition"] for row in raw] == [
        "APPLIED",
        "APPLIED",
        "APPLIED",
        "CONFLICT",
        "REJECTED",
        "APPLIED",
        "APPLIED",
        "APPLIED",
        "APPLIED",
        "APPLIED",
        "DUPLICATE",
    ]
    event_count_before_restart = replay.snapshot().event_count
    replay.close()

    restarted = NautilusOrderReplay(database)
    assert restarted.snapshot().status == "FILLED"
    assert restarted.snapshot().filled_quantity_raw == 100
    assert restarted.snapshot().event_count == event_count_before_restart
    assert restarted.snapshot().event_count == 9
    assert len(restarted.applied_events()) == 8
    assert len(restarted.raw_observations()) == 11

    path = [
        "SUBMISSION_RESERVED",
        "ACKNOWLEDGED",
        "PARTIALLY_FILLED",
        "CANCEL_REQUESTED",
        "CANCELED",
        "FILLED",
    ]
    previous = None
    for state in path:
        assert state in TRANSITIONS[previous]
        previous = state

    expected = (
        fill_contract(partial, "TRADE-PARTIAL", 40),
        fill_contract(late, "TRADE-LATE", 60),
    )
    ledger = Ledger(
        account="SIM-AXIOM2",
        mode="observed",
        initial_cash_cents=200_000,
        initial_positions=(("AAPL.SIM", 0),),
        known_orders=("O-AXIOM2-REPLAY",),
        expected_fills=expected,
        peak_equity_cents=200_000,
        external_cashflow_cents=0,
    )
    observations = Observations(
        account="SIM-AXIOM2",
        mode="observed",
        complete=True,
        fills=expected,
    )
    account = Account(
        account="SIM-AXIOM2",
        total_cash_cents=99_980,
        settled_cash_cents=99_980,
        positions=(("AAPL.SIM", 100),),
        complete=True,
        equity_cents=199_980,
        external_cashflow_cents=0,
        valuation_marks_micros=(("AAPL.SIM", 10_000_000),),
    )
    decision = reconcile(
        ledger,
        observations,
        account,
        Policy(policy_digest="a" * 64, cash_tolerance_cents=0, fee_tolerance_cents=0),
    )
    assert decision.state == "MATCHED"

    conflicted_observations = Observations(
        account="SIM-AXIOM2",
        mode="observed",
        complete=True,
        fills=(
            expected[0],
            fill_contract(conflict, "TRADE-PARTIAL", 40),
            expected[1],
        ),
    )
    assert (
        reconcile(
            ledger,
            conflicted_observations,
            account,
            Policy(
                policy_digest="a" * 64,
                cash_tolerance_cents=0,
                fee_tolerance_cents=0,
            ),
        ).state
        == "HALTED"
    )


def test_projection_consumes_actual_reports_and_reconciles(tmp_path):
    replay = NautilusOrderReplay(tmp_path/'projection.sqlite')
    replay.ingest('submit', submitted(1), received_at_ns=1)
    replay.ingest('ack', accepted(2), received_at_ns=2)
    replay.ingest('partial', filled(3, 'T1', 40), received_at_ns=3)
    projection = replay.projection()
    assert projection.lifecycle_state == 'PARTIALLY_FILLED'
    assert projection.unresolved == ()
    assert projection.fills[0].quantity == 40
    assert projection.fills[0].fee_cents == 10
    replay.ingest('cancel', canceled(4), received_at_ns=4)
    replay.ingest('late', filled(5, 'T2', 60), received_at_ns=5)
    projection = replay.projection()
    assert projection.lifecycle_state == 'FILLED'
    assert projection.unresolved == ()
    ledger = Ledger('SIM-AXIOM2', 'observed', 200_000, (('AAPL.SIM',0),),
        ('O-AXIOM2-REPLAY',), projection.fills, 200_000, 0)
    account = Account('SIM-AXIOM2', 99_980, 99_980, (('AAPL.SIM',100),), True,
        199_980, 0, valuation_marks_micros=(('AAPL.SIM',10_000_000),))
    assert replay.compare_reconciliation(ledger, account,
        Policy('a'*64, 0, 0)).state == 'MATCHED'
    replay.ingest('partial', filled(6, 'T1', 40, price='10.01'), received_at_ns=6)
    assert replay.projection().unresolved
    assert replay.compare_reconciliation(ledger, account,
        Policy('a'*64, 0, 0)).state != 'MATCHED'


def test_order_seed_binding_and_raw_integrity_survive_restart(tmp_path):
    from dataclasses import replace
    import pytest
    from axiom2.nautilus_runtime import OrderSeed, NautilusOrderReplayCorruption
    path = tmp_path/'seed.sqlite'
    replay = NautilusOrderReplay(path)
    replay.ingest('submit', submitted(1), received_at_ns=1)
    replay.close()
    with pytest.raises(NautilusOrderReplayCorruption, match='seed'):
        NautilusOrderReplay(path, seed=replace(OrderSeed(), quantity=200))
    replay = NautilusOrderReplay(path)
    replay.connection.execute("UPDATE raw_order_observations SET payload_json='{}'")
    replay.close()
    with pytest.raises(NautilusOrderReplayCorruption, match='digest'):
        NautilusOrderReplay(path)


def test_two_connections_rebuild_before_applying_report(tmp_path):
    path = tmp_path/'two.sqlite'
    first = NautilusOrderReplay(path)
    second = NautilusOrderReplay(path)
    first.ingest('submit', submitted(1), received_at_ns=1)
    second.ingest('ack', accepted(2), received_at_ns=2)
    first.ingest('partial', filled(3, 'T1', 40), received_at_ns=3)
    second.ingest('rest', filled(4, 'T2', 60), received_at_ns=4)
    assert first.snapshot().status == 'FILLED'
    assert second.snapshot().filled_quantity_raw == 100


def test_event_identity_dedupes_across_receipt_identities(tmp_path):
    replay = NautilusOrderReplay(tmp_path/'ids.sqlite')
    event = submitted(1)
    replay.ingest('first', event, received_at_ns=1)
    assert replay.ingest('second', event, received_at_ns=2).disposition == 'DUPLICATE'
    assert replay.snapshot().event_count == 2
    assert len(replay.raw_observations()) == 2
