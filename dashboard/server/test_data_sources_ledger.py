import json
from datetime import datetime, timedelta, timezone

from dashboard.server import data_sources as ds


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_read_trades_surfaces_order_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "OANDA_DIR", tmp_path)
    _write_jsonl(tmp_path / "transactions.jsonl", [
        {
            "id": "10",
            "time": "2026-06-30T10:00:00Z",
            "type": "MARKET_ORDER",
            "instrument": "EUR_USD",
            "units": "1000",
            "reason": "CLIENT_ORDER",
            "clientExtensions": {"tag": "ml_engine_trend_demo"},
        },
        {
            "id": "11",
            "time": "2026-06-30T10:00:00Z",
            "type": "ORDER_FILL",
            "orderID": "10",
            "instrument": "EUR_USD",
            "units": "1000",
            "price": "1.10123",
            "pl": "0.0000",
            "tradeOpened": {"tradeID": "11"},
            "accountBalance": "100000.00",
        },
        {
            "id": "12",
            "time": "2026-06-30T10:00:00Z",
            "type": "STOP_LOSS_ORDER",
            "tradeID": "11",
            "price": "1.09500",
            "reason": "ON_FILL",
        },
        {
            "id": "13",
            "time": "2026-06-30T11:00:00Z",
            "type": "ORDER_CANCEL",
            "orderID": "12",
            "reason": "LINKED_TRADE_CLOSED",
        },
        {
            "id": "14",
            "time": "2026-06-30T11:01:00Z",
            "type": "MARKET_ORDER_REJECT",
            "instrument": "EUR_USD",
            "units": "-1000",
            "rejectReason": "INSUFFICIENT_MARGIN",
        },
    ])

    out = ds.read_trades(limit=20)
    rows = {row["id"]: row for row in out["trades"]}

    assert out["count"] == 5
    assert out["fill_count"] == 1
    assert rows["10"]["status"] == "FILLED"
    assert rows["10"]["linked_fill_id"] == "11"
    assert rows["10"]["linked_fill_price"] == 1.10123
    assert rows["11"]["status"] == "FILLED"
    assert rows["11"]["fill_kind"] == "OPEN"
    assert rows["11"]["order_id"] == "10"
    assert rows["11"]["trade_ids"] == ["11"]
    assert rows["12"]["status"] == "CANCELLED"
    assert rows["12"]["instrument"] == "EUR_USD"
    assert rows["12"]["trade_ids"] == ["11"]
    assert rows["13"]["status"] == "CANCELLED"
    assert rows["14"]["status"] == "REJECTED"


def test_read_equity_uses_balance_rows_and_fill_realized_pl(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "OANDA_DIR", tmp_path)
    _write_jsonl(tmp_path / "transactions.jsonl", [
        {
            "id": "20",
            "time": "2026-06-30T10:00:00Z",
            "type": "ORDER_FILL",
            "pl": "7.50",
            "accountBalance": "100007.50",
        },
        {
            "id": "21",
            "time": "2026-06-30T12:00:00Z",
            "type": "DAILY_FINANCING",
            "accountBalance": "100006.25",
        },
    ])

    out = ds.read_equity()

    assert out["n"] == 2
    assert out["points"][-1]["balance"] == 100006.25
    assert out["ledger_realized_pl"] == 7.5


def test_read_crypto_momentum_empty_ledger_does_not_raise(tmp_path, monkeypatch):
    ledger_path = tmp_path / "shadow_momentum_ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(ds, "_CRYPTO_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(ds, "_CRYPTO_LIVE_GATE_STATE_PATH", tmp_path / "live_gate_state.json")

    out = ds.read_crypto_momentum()

    assert out["has_run"] is False
    assert out["n_forward_cycles"] == 0
    assert out["forward_sharpe_annualized"] is None
    assert out["current_book"] is None
    assert out["cumulative_shadow_return"] == 0.0


def test_read_crypto_momentum_populated_ledger_surfaces_last_cycle(tmp_path, monkeypatch):
    ledger_path = tmp_path / "shadow_momentum_ledger.jsonl"
    _write_jsonl(ledger_path, [
        {
            "asof_date": "2026-06-29",
            "book": {"BTC": 0.5, "ETH": -0.5},
            "gross_leverage": 1.0,
            "cumulative_shadow_return": 0.012,
            "today_net_return": 0.012,
            "construction": {"quintile": 5},
        },
        {
            "asof_date": "2026-06-30",
            "book": {"BTC": 0.4, "ETH": -0.6},
            "gross_leverage": 1.0,
            "cumulative_shadow_return": 0.018,
            "today_net_return": 0.006,
            "construction": {"quintile": 5},
        },
    ])
    monkeypatch.setattr(ds, "_CRYPTO_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(ds, "_CRYPTO_LIVE_GATE_STATE_PATH", tmp_path / "live_gate_state.json")

    out = ds.read_crypto_momentum()

    assert out["has_run"] is True
    assert out["n_forward_cycles"] == 2
    assert out["current_asof"] == "2026-06-30"
    assert out["current_book"] == {"BTC": 0.4, "ETH": -0.6}
    assert out["cumulative_shadow_return"] == 0.018
    assert out["forward_sharpe_annualized"] is not None


def test_read_crypto_carry_empty_ledger_does_not_raise(tmp_path, monkeypatch):
    ledger_path = tmp_path / "shadow_carry_ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(ds, "_CRYPTO_CARRY_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(ds, "_CRYPTO_CARRY_LIVE_GATE_STATE_PATH", tmp_path / "live_gate_state.json")

    out = ds.read_crypto_carry()

    assert out["has_run"] is False
    assert out["n_forward_cycles"] == 0
    assert out["forward_sharpe_annualized"] is None
    assert out["current_book"] is None
    assert out["cumulative_shadow_return"] == 0.0
    assert out["risk_premium_note"] is not None  # always present, even on an empty ledger


def test_read_crypto_carry_populated_ledger_surfaces_last_cycle(tmp_path, monkeypatch):
    ledger_path = tmp_path / "shadow_carry_ledger.jsonl"
    _write_jsonl(ledger_path, [
        {
            "asof_date": "2026-06-29",
            "book": {"longs": {"BTCUSDT": 0.05}, "shorts": {}},
            "gross_leverage": 1.0,
            "cumulative_shadow_return": 0.004,
            "today_net_return": 0.004,
            "construction": {"cost_bps_roundtrip": 20.0},
        },
        {
            "asof_date": "2026-06-30",
            "book": {"longs": {"BTCUSDT": 0.04, "ETHUSDT": 0.04}, "shorts": {}},
            "gross_leverage": 1.0,
            "cumulative_shadow_return": 0.006,
            "today_net_return": 0.002,
            "construction": {"cost_bps_roundtrip": 20.0},
        },
    ])
    monkeypatch.setattr(ds, "_CRYPTO_CARRY_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(ds, "_CRYPTO_CARRY_LIVE_GATE_STATE_PATH", tmp_path / "live_gate_state.json")

    out = ds.read_crypto_carry()

    assert out["has_run"] is True
    assert out["n_forward_cycles"] == 2
    assert out["current_asof"] == "2026-06-30"
    assert out["current_book"] == {"longs": {"BTCUSDT": 0.04, "ETHUSDT": 0.04}, "shorts": {}}
    assert out["cumulative_shadow_return"] == 0.006
    assert out["forward_sharpe_annualized"] is not None
    assert out["live_gate"]["armed"] is False  # no live_gate_state.json on disk -> never armed


def test_shadow_lane_freshness_uses_writer_timestamp_not_book_asof(tmp_path, monkeypatch):
    ledger_path = tmp_path / "shadow_carry_ledger.jsonl"
    _write_jsonl(ledger_path, [{
        "asof_date": "2026-06-01",
        "cycle_ts": "2026-07-10T00:00:00Z",
        "book": {"longs": {}, "shorts": {}},
        "construction": {"rebalance_days": 1},
    }])
    monkeypatch.setattr(ds, "_CRYPTO_CARRY_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(ds, "_CRYPTO_CARRY_LIVE_GATE_STATE_PATH", tmp_path / "live_gate_state.json")
    monkeypatch.setattr(ds.time, "time", lambda: datetime(2026, 7, 14, tzinfo=timezone.utc).timestamp())

    out = ds.read_crypto_carry()

    assert out["freshness"]["last_evaluated_at"] == "2026-07-10T00:00:00Z"
    assert out["freshness"]["age_s"] == 4 * 86_400
    assert out["freshness"]["expected_interval_s"] == 86_400
    assert out["freshness"]["status"] == "stale"




def test_read_tier7_stale_snapshot_cannot_report_running(tmp_path, monkeypatch):
    state_path = tmp_path / ".claude" / "tier7_state.json"
    state_path.parent.mkdir(parents=True)
    generated_at = (datetime.now(timezone.utc) - timedelta(seconds=ds.TIER7_FRESH_S + 5)).isoformat()
    state_path.write_text(json.dumps({
        "generated_at": generated_at,
        "running": True,
        "running_reason": "heartbeat fresh + pid alive",
        "autonomy_level": "L5",
        "max_autonomy": 5,
        "bounded": True,
        "last_cycle": {"pid": 123, "pid_alive": True},
    }), encoding="utf-8")
    monkeypatch.setattr(ds, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ds, "TIER7_STATE_PATH", state_path)

    out = ds.read_tier7()

    assert out["connected"] is True
    assert out["snapshot_stale"] is True
    assert out["running"] is False
    assert out["running_reason"].startswith("snapshot stale")
