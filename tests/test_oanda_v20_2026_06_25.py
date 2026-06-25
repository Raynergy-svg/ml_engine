"""Robust OANDA v20 streaming + ledger layer — pure-logic tests (no network, no mocks).

Stream line-parsing, reconnect/backoff, practice-host pinning, and transaction-
ledger realized-P&L aggregation are all exercised against synthetic data + real
disk (tmp_path). The live HTTP stream is integration (real practice token), not here.
"""
from __future__ import annotations

import json
import threading

import pytest

from src.brokers.oanda_v20 import (
    MAX_BACKOFF_S,
    OandaV20Error,
    TransactionLedger,
    _assert_practice,
    _backoff_seq,
    consume_stream,
    run_stream,
)


def test_assert_practice_refuses_live_host():
    _assert_practice("https://stream-fxpractice.oanda.com/v3/accounts/x/pricing/stream")  # ok
    with pytest.raises(OandaV20Error):
        _assert_practice("https://stream-fxtrade.oanda.com/v3/accounts/x/pricing/stream")
    with pytest.raises(OandaV20Error):
        _assert_practice("https://api-fxtrade.oanda.com/v3/x")


def test_consume_stream_parses_and_drops_heartbeats():
    seen = []
    lines = [
        json.dumps({"type": "PRICE", "instrument": "EUR_USD", "bids": [{"price": "1.1"}]}),
        json.dumps({"type": "HEARTBEAT", "time": "t"}),                 # dropped
        b'{"type": "PRICE", "instrument": "USD_JPY"}',                  # bytes ok
        "",                                                             # blank skipped
        "{not json}",                                                   # malformed skipped
        json.dumps({"type": "ORDER_FILL", "pl": "3.5"}),
    ]
    n = consume_stream(lines, on_message=seen.append)
    assert n == 3
    assert [m.get("instrument") or m.get("type") for m in seen] == ["EUR_USD", "USD_JPY", "ORDER_FILL"]


def test_backoff_caps_and_floors():
    seq = _backoff_seq(0.6)
    vals = [next(seq) for _ in range(12)]
    assert vals[0] >= 0.6
    assert vals[-1] == MAX_BACKOFF_S
    assert all(b <= MAX_BACKOFF_S for b in vals)


def test_run_stream_reconnects_then_stops():
    # First open raises (forces a reconnect); second yields one message; then we stop.
    stop = threading.Event()
    calls = {"n": 0}
    reconnects = []

    def open_conn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("boom")
        return [json.dumps({"type": "PRICE", "instrument": "EUR_USD"})]

    def on_message(_obj):
        stop.set()  # got our message; end the loop

    run_stream(open_conn, on_message, stop=stop,
               on_reconnect=lambda d, e: reconnects.append((d, type(e).__name__)))
    assert calls["n"] >= 2                     # reconnected after the first failure
    assert reconnects and reconnects[0][1] == "ConnectionError"


def test_transaction_ledger_realized_pl(tmp_path):
    # Real disk: write a synthetic ledger jsonl, confirm realized P&L + financing sum.
    led_dir = tmp_path / "oanda"
    led_dir.mkdir()
    (led_dir / "transactions.jsonl").write_text("\n".join([
        json.dumps({"id": "10", "type": "ORDER_FILL", "pl": "12.5", "financing": "-0.2"}),
        json.dumps({"id": "11", "type": "DAILY_FINANCING", "financing": "-1.0"}),
        json.dumps({"id": "12", "type": "ORDER_FILL", "pl": "-4.0", "financing": "0.1"}),
    ]) + "\n")
    (led_dir / "ledger_state.json").write_text(json.dumps({"last_transaction_id": "12"}))

    led = TransactionLedger(client=None, ledger_dir=led_dir)
    s = led.summarize()
    assert s.n_transactions == 3
    assert s.n_fills == 2
    assert abs(s.realized_pl - 8.5) < 1e-9        # 12.5 + (-4.0)
    assert abs(s.financing - (-0.1)) < 1e-9       # -0.2 + 0.1 (fill financing only)
    assert s.last_transaction_id == "12"
