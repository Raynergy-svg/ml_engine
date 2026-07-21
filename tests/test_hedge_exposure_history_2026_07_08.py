"""No-mock tests for src/hedge/exposure_history.py — real files on tmp_path,
real parquet tick fixtures, the real committed bucket maps. No unittest.mock
anywhere (repo No-Mock Rule, 2026-05-01)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.hedge.exposure_history import capture_exposure_snapshot

NOW = datetime.now(timezone.utc)


def _write_account_state(path: Path, positions, *, nav=100000.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "account_id": "101-000-0000000-001", "currency": "USD",
        "nav": nav, "margin_used": 5000.0, "margin_available": 95000.0,
        "open_trade_count": len(positions), "unrealized_pl": 12.5,
        "realized_pl": 0.0, "positions": positions,
    }
    path.write_text(json.dumps(state), encoding="utf-8")


def _write_ticks(root: Path, pair: str, mid: float, *, ts: datetime = None) -> None:
    ts = ts or NOW
    f = root / pair / f"{ts:%Y}" / f"{ts:%m}" / f"{ts:%d}.parquet"
    f.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.DatetimeIndex([ts - timedelta(seconds=30), ts], name="time")
    df = pd.DataFrame({"instrument": pair, "bid": mid - 0.0001, "ask": mid + 0.0001,
                       "mid": [mid, mid], "bid_liq": 1e6, "ask_liq": 1e6,
                       "status": "tradeable", "source": "test"}, index=idx)
    df.to_parquet(f)


@pytest.fixture
def env(tmp_path):
    state = tmp_path / "oanda" / "account_state.json"
    ticks = tmp_path / "ticks"
    out = tmp_path / "hedge" / "exposure_history.jsonl"
    return state, ticks, out


def _capture(state, ticks, out, **kw):
    return capture_exposure_snapshot(account_state_path=state, ticks_root=ticks,
                                     out_path=out, now=NOW, **kw)


def test_happy_path_long_usdjpy_short_eurusd(env):
    state, ticks, out = env
    _write_account_state(state, [
        {"instrument": "USD_JPY", "net_units": 100000.0, "stop_loss": 160.0,
         "unrealized_pl": 50.0},
        {"instrument": "EUR_USD", "net_units": -50000.0, "stop_loss": 1.10,
         "unrealized_pl": -20.0},
    ])
    _write_ticks(ticks, "USD_JPY", 162.5)
    _write_ticks(ticks, "EUR_USD", 1.08)

    row = _capture(state, ticks, out)
    assert row is not None
    net = row["net_currency_notional_home"]
    # Long USD_JPY 100k = +100k USD / -100k-equivalent JPY;
    # short EUR_USD 50k EUR (54k USD notional) = -54k EUR / +54k USD.
    assert net["USD"] == pytest.approx(100000.0 + 54000.0)
    assert net["JPY"] == pytest.approx(-100000.0)
    assert net["EUR"] == pytest.approx(-54000.0)
    # R-based risk: 100k * |162.5-160.0| * (1/162.5) quote(JPY)->USD.
    usdjpy = next(p for p in row["positions"] if p["instrument"] == "USD_JPY")
    assert usdjpy["risk_home_r"] == pytest.approx(100000 * 2.5 / 162.5)
    assert usdjpy["direction"] == "long"
    eurusd = next(p for p in row["positions"] if p["instrument"] == "EUR_USD")
    assert eurusd["direction"] == "short"
    assert eurusd["risk_home_r"] == pytest.approx(50000 * 0.02)  # quote USD -> 1.0
    # Correlation buckets are the fx_currency:-namespaced same nets.
    assert row["net_correlation_buckets_notional_home"]["fx_currency:USD"] == \
        pytest.approx(net["USD"])
    assert row["fail_reasons"] == []
    assert row["paper_only"] is True and row["runtime_allowed"] is False
    # Row really persisted.
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["nav"] == pytest.approx(100000.0)


def test_missing_account_state_refuses(env):
    state, ticks, out = env
    assert _capture(state, ticks, out) is None
    assert not out.exists()


def test_stale_account_state_refuses(env):
    state, ticks, out = env
    _write_account_state(state, [])
    old = (NOW - timedelta(hours=3)).timestamp()
    os.utime(state, (old, old))
    assert _capture(state, ticks, out) is None
    assert not out.exists()


def test_corrupt_account_state_refuses(env):
    state, ticks, out = env
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("{not json", encoding="utf-8")
    assert _capture(state, ticks, out) is None


def test_flat_book_appends_zero_exposure_row(env):
    state, ticks, out = env
    _write_account_state(state, [])
    row = _capture(state, ticks, out)
    assert row is not None
    assert row["positions"] == []
    assert row["net_currency_notional_home"] == {}


def test_unresolvable_rate_fails_closed_not_zeroed(env):
    state, ticks, out = env
    _write_account_state(state, [
        {"instrument": "USD_JPY", "net_units": 100000.0, "stop_loss": 160.0},
        {"instrument": "EUR_GBP", "net_units": 10000.0, "stop_loss": 0.84},
    ])
    _write_ticks(ticks, "USD_JPY", 162.5)  # no EUR rate anywhere -> EUR_GBP fails
    row = _capture(state, ticks, out)
    assert row is not None
    assert row["skipped_instruments"] == ["EUR_GBP"]
    assert any("no_base_to_home_rate:EUR_GBP" in r for r in row["fail_reasons"])
    # The resolved position still netted; the failed one contributed NOTHING.
    assert row["net_currency_notional_home"]["USD"] == pytest.approx(100000.0)
    assert "GBP" not in row["net_currency_notional_home"]


def test_all_positions_unresolvable_refuses_row(env):
    state, ticks, out = env
    _write_account_state(state, [
        {"instrument": "EUR_GBP", "net_units": 10000.0, "stop_loss": 0.84},
    ])
    assert _capture(state, ticks, out) is None  # no ticks at all
    assert not out.exists()


def test_dedupe_on_source_mtime(env):
    state, ticks, out = env
    _write_account_state(state, [])
    assert _capture(state, ticks, out) is not None
    assert _capture(state, ticks, out) is None  # same mtime -> refused
    assert len(out.read_text().strip().splitlines()) == 1
    # A NEW snapshot (mtime bump) appends again.
    _write_account_state(state, [])
    future = (NOW + timedelta(seconds=5)).timestamp()
    os.utime(state, (future, future))
    assert _capture(state, ticks, out) is not None
    assert len(out.read_text().strip().splitlines()) == 2


def test_stale_ticks_do_not_supply_rates(env):
    state, ticks, out = env
    _write_account_state(state, [
        {"instrument": "EUR_USD", "net_units": 50000.0, "stop_loss": 1.05},
    ])
    _write_ticks(ticks, "EUR_USD", 1.08, ts=NOW - timedelta(hours=2))
    assert _capture(state, ticks, out) is None  # only position unresolvable -> refuse


def test_missing_stop_loss_records_reason_but_keeps_notional(env):
    state, ticks, out = env
    _write_account_state(state, [
        {"instrument": "USD_CAD", "net_units": 80000.0},
    ])
    _write_ticks(ticks, "USD_CAD", 1.36)
    row = _capture(state, ticks, out)
    assert row is not None
    pos = row["positions"][0]
    assert pos["notional_home"] == pytest.approx(80000.0)  # base USD -> 1.0
    assert pos["risk_home_r"] is None
    assert any("no_stop_loss" in r for r in pos["fail_reasons"])


def test_no_execution_path_imports():
    """Structural: the capture module must never import the execution plane
    (same import-boundary discipline as the risk-target pre-registration)."""
    root = Path(__file__).resolve().parents[1]
    for rel in ("src/hedge/exposure_history.py",
                "scripts/run_exposure_history_capture.py"):
        src = (root / rel).read_text(encoding="utf-8")
        for banned in ("scanner.execution", "state_engine", "oanda_practice",
                       "brokers", "execute_trade", "place_order"):
            assert banned not in src, f"execution-plane reference in {rel}: {banned}"


def test_append_failure_reported_as_refusal(env):
    """G1: if the JSONL append cannot persist (out_path's parent is an
    unwritable location — here: a FILE where the dir should be), the capture
    must return None, not a phantom row."""
    state, ticks, out_dirfile = env
    _write_account_state(state, [])
    blocked_parent = state.parent / "blocked"
    blocked_parent.write_text("a file, not a directory", encoding="utf-8")
    out = blocked_parent / "exposure_history.jsonl"  # mkdir will fail
    assert capture_exposure_snapshot(account_state_path=state, ticks_root=ticks,
                                     out_path=out, now=NOW) is None
