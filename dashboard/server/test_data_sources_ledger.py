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
