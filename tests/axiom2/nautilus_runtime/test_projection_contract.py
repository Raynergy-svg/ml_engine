"""Hand-authored normalized projection fixtures; native parity is tested separately."""
import hashlib
import json
from decimal import localcontext

import pytest
from src.axiom2.nautilus_runtime.order_replay import OrderSeed, project_order_rows


def row(kind, status, **fields):
    payload = dict(type=kind, trader_id='TRADER-AXIOM2-REPLAY', strategy_id='S-AXIOM2-REPLAY',
        instrument_id='AAPL.SIM', client_order_id='O-AXIOM2-REPLAY', account_id='SIM-AXIOM2',
        ts_event=1, **fields)
    serialized = json.dumps(payload, sort_keys=True)
    return dict(event_type=kind, observation_id=kind, disposition='APPLIED', payload_json=serialized,
        payload_digest=hashlib.sha256(serialized.encode()).hexdigest(), snapshot_json=json.dumps({'status':status}))


def reports(**fill_fields):
    return (row('OrderSubmitted','SUBMITTED'), row('OrderAccepted','ACCEPTED'),
        row('OrderFilled','FILLED', trade_id='T1', order_side='BUY', order_type='LIMIT',
            last_qty='100', last_px='10.00', currency='USD', commission='0.10 USD', **fill_fields))


def test_exact_projection_money_is_independent_of_decimal_context():
    with localcontext() as context:
        context.prec = 2
        result = project_order_rows(OrderSeed(), reports())
    assert result.lifecycle_state == 'FILLED'
    assert result.fills[0].quantity == 100
    assert result.fills[0].price_cents == 1000
    assert result.fills[0].fee_cents == 10
    assert result.unresolved == ()
    assert result.execution_enabled is False
    assert result.capital_authorized is False


@pytest.mark.parametrize('field,value', [('last_px','10.001'), ('last_qty','0.5'), ('commission',None), ('currency','EUR')])
def test_unsupported_fill_facts_remain_unresolved(field, value):
    rows = list(reports())
    payload = json.loads(rows[-1]['payload_json'])
    payload[field] = value
    serialized = json.dumps(payload, sort_keys=True)
    rows[-1]['payload_json'] = serialized
    rows[-1]['payload_digest'] = hashlib.sha256(serialized.encode()).hexdigest()
    result = project_order_rows(OrderSeed(), rows)
    assert result.unresolved
    assert result.fills == ()


def test_cancel_rejection_with_no_fill_records_axiom_parity_gap():
    rows = (row('OrderSubmitted','SUBMITTED'), row('OrderAccepted','ACCEPTED'),
        row('OrderPendingCancel','PENDING_CANCEL'), row('OrderCancelRejected','ACCEPTED'))
    assert project_order_rows(OrderSeed(), rows).unresolved


def test_fill_above_seed_bound_remains_unresolved():
    rows = list(reports())
    payload = json.loads(rows[-1]['payload_json'])
    payload['last_qty'] = '101'
    serialized = json.dumps(payload, sort_keys=True)
    rows[-1]['payload_json'] = serialized
    rows[-1]['payload_digest'] = hashlib.sha256(serialized.encode()).hexdigest()
    assert project_order_rows(OrderSeed(), rows).unresolved
