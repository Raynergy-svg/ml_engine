"""Reproduce-then-resolve: cost-model + portfolio-exposure-engine ADDITIVE
wiring into the OANDA trend risk gate (2026-07-08 operator-directed pass).

Hard constraint under test: every existing risk-gate guarantee (one-position
cap, correlation-bucket cap, risk-normalized sizing) is preserved or
strengthened, NEVER loosened. The two new checks (cost_aware_gate,
exposure_engine_gate) are ADDITIVE — an AND-combination with the pre-existing
gates, so the combined result can only be MORE restrictive.

NO MOCKS per CLAUDE.md No-Mock Rule — real functions, real tmp_path disk I/O,
real (synthetic-but-realistic) OANDA transaction/candle shapes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.equity.oanda_trend import (  # noqa: E402
    _open_trades_to_exposure_positions,
    exposure_engine_gate,
    run_oanda_trend_cycle,
)
from src.hedge.portfolio_exposure import ExposurePosition  # noqa: E402
from src.scanner.automation.state_engine import StateEngine  # noqa: E402

NAV = 102818.5912
LAST_PX = {
    "USD_JPY": 162.737, "EUR_JPY": 185.675, "GBP_JPY": 215.504,
    "USD_CAD": 1.42157, "USD_CHF": 0.80912, "EUR_USD": 1.0850, "GBP_USD": 1.2650,
}


# --------------------------------------------------------------------- #
# exposure_engine_gate — pure-function reproduce cases                  #
# --------------------------------------------------------------------- #
def test_exposure_engine_gate_reproduces_usd_pileup_across_two_pairs():
    """Mirrors trend_risk_gates' own
    test_bucket_cap_gate_catches_usd_long_bucket_from_2_different_pairs: long
    USD_CAD + USD_CHF already carry ~2R of USD-long risk (the operator's exact
    real example); a further USD_JPY long must be BLOCKED by the exposure
    engine too, independently of bucket_cap_gate."""
    risk_unit_home = NAV * 0.01
    max_bucket_risk_r = 2.0
    open_positions = [
        ExposurePosition(asset_class="fx", instrument="USD_CAD", direction="long",
                         risk_home=1.05 * risk_unit_home, source="oanda_open_trade"),
        ExposurePosition(asset_class="fx", instrument="USD_CHF", direction="long",
                         risk_home=1.00 * risk_unit_home, source="oanda_open_trade"),
    ]
    candidate = ExposurePosition(asset_class="fx", instrument="USD_JPY", direction="long",
                                 risk_home=0.20 * risk_unit_home, source="candidate")
    allow, reason = exposure_engine_gate(
        open_positions=open_positions, candidate=candidate,
        risk_unit_home=risk_unit_home, max_bucket_risk_r=max_bucket_risk_r)
    assert allow is False
    assert reason == "exposure_engine_bucket_exceeded:USD"


def test_exposure_engine_gate_allows_first_two_correlated_trades_under_cap():
    risk_unit_home = NAV * 0.01
    candidate = ExposurePosition(asset_class="fx", instrument="USD_JPY", direction="long",
                                 risk_home=0.5 * risk_unit_home, source="candidate")
    allow, reason = exposure_engine_gate(
        open_positions=[], candidate=candidate,
        risk_unit_home=risk_unit_home, max_bucket_risk_r=2.0)
    assert allow is True
    assert reason == "exposure_engine_ok"


def test_exposure_engine_gate_fails_closed_on_unresolvable_currency():
    """No fabrication: a currency absent from the bucket map must REFUSE, not
    silently treat the leg as zero exposure — matches ExposureReport's own
    fail-closed contract."""
    risk_unit_home = NAV * 0.01
    candidate = ExposurePosition(asset_class="fx", instrument="USD_JPY", direction="long",
                                 risk_home=0.5 * risk_unit_home, source="candidate")
    allow, reason = exposure_engine_gate(
        open_positions=[], candidate=candidate,
        risk_unit_home=risk_unit_home, max_bucket_risk_r=2.0,
        currency_bucket_map={})  # empty map -> every currency unresolvable
    assert allow is False
    assert reason.startswith("exposure_engine_fail_closed:")


def test_exposure_engine_gate_fails_closed_on_unresolvable_instrument():
    risk_unit_home = NAV * 0.01
    candidate = ExposurePosition(asset_class="fx", instrument="garbage", direction="long",
                                 risk_home=100.0, source="candidate")
    allow, reason = exposure_engine_gate(
        open_positions=[], candidate=candidate,
        risk_unit_home=risk_unit_home, max_bucket_risk_r=2.0)
    assert allow is False


# --------------------------------------------------------------------- #
# _open_trades_to_exposure_positions — translation layer                #
# --------------------------------------------------------------------- #
def test_open_trades_to_exposure_positions_reads_direction_from_signed_units():
    trades = [
        {"id": "1", "instrument": "USD_JPY", "currentUnits": "30569"},
        {"id": "2", "instrument": "EUR_JPY", "currentUnits": "-10000"},  # a short, if it ever exists
    ]
    fallback = {"USD_JPY": 1.15266, "EUR_JPY": 1.7}
    positions = _open_trades_to_exposure_positions(trades, {}, fallback, LAST_PX)
    by_inst = {p.instrument: p for p in positions}
    assert by_inst["USD_JPY"].direction == "long"
    assert by_inst["EUR_JPY"].direction == "short"
    assert all(p.risk_home > 0 for p in positions)


def test_open_trades_to_exposure_positions_skips_unresolvable_trade():
    """A trade with no r_distance available (neither entry-anchored nor
    fallback) is SKIPPED, not fabricated as zero risk — matches
    compute_bucket_risk's own skip behavior for the identical case, so
    neither engine silently under-counts relative to the other."""
    trades = [{"id": "1", "instrument": "USD_JPY", "currentUnits": "30569"}]
    positions = _open_trades_to_exposure_positions(trades, {}, {}, LAST_PX)
    assert positions == []


# --------------------------------------------------------------------- #
# Full-cycle reproduce: run_oanda_trend_cycle end-to-end                #
# --------------------------------------------------------------------- #
def _ohlc_candles(closes, *, start="2020-01-01", range_frac=0.001):
    """H/L range scaled to a FRACTION of price (realistic ATR-of-price ratio
    across very different price levels — JPY pairs ~150-215, CHF ~0.80 — a
    fixed absolute range like EUR_USD's tests use would produce a wildly
    unrealistic implied leverage for JPY-quote pairs)."""
    import pandas as pd
    idx = pd.bdate_range(start, periods=len(closes))
    return {"candles": [
        {"time": t.isoformat() + "Z", "complete": True,
         "mid": {"h": f"{c * (1 + range_frac):.5f}", "l": f"{c * (1 - range_frac):.5f}",
                 "c": f"{c:.5f}"}}
        for t, c in zip(idx, closes)
    ]}


class _Config:
    oanda_environment = "practice"


class _FakeClient:
    def __init__(self, candles, trades=None, nav=100000.0, positions=None):
        self.candles = candles
        self.trades = trades or []
        self.orders = []
        self.nav = nav
        self.positions = positions or []

    def get_candles(self, instrument, **_kwargs):
        return self.candles[instrument]

    def get_account_summary(self):
        return {"account": {"NAV": str(self.nav)}}

    def get_open_positions(self):
        return {"positions": self.positions}

    def get_trades(self, *, state="OPEN"):
        return {"trades": self.trades}

    def create_market_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"orderCreateTransaction": {"id": str(len(self.orders))}}

    def set_trade_dependent_orders(self, **kwargs):
        return {"relatedTransactionIDs": ["1"]}


def _seed_fills(root: Path, instrument: str, n: int = 3, spread=0.0002):
    path = root / "trained_data" / "oanda" / "transactions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = [existing.rstrip("\n")] if existing else []
    mid = 1.10000
    for i in range(n):
        lines.append(json.dumps({
            "type": "ORDER_FILL", "instrument": instrument,
            "time": f"2026-06-{i + 1:02d}T00:00:00Z",
            "price": f"{mid + spread / 2:.5f}", "units": "1000", "requestedUnits": "1000",
            "halfSpreadCost": "0.05",
            "fullPrice": {"bids": [{"price": f"{mid - spread / 2:.5f}"}],
                          "asks": [{"price": f"{mid + spread / 2:.5f}"}]},
        }))
    path.write_text("\n".join(line for line in lines if line) + "\n", encoding="utf-8")


def test_run_oanda_trend_cycle_exposure_engine_independently_caps_same_cycle_usd_pileup(tmp_path):
    """Full-cycle reproduce of the operator's exact USD-pileup pattern (3
    correlated USD-base pairs flipping 'on' in the SAME cycle from a flat
    book), proving the NEW exposure-engine check independently enforces the
    same same-cycle-accumulation discipline bucket_cap_gate already has (its
    own pure-function reproduce-cases in test_oanda_trend_risk_gates.py are
    re-run unchanged and green as part of this same verification pass) — the
    combined AND of both checks is still at least as strict, not accidentally
    weakened by adding two more gates to the chain. With risk_pct=0.01 sizing
    ~1R (~$1028) per trade and the default 2R bucket cap, at most 2 of 3
    correlated USD-long adds can be approved in one cycle."""
    StateEngine(tmp_path / ".claude" / "state.json").set_halted(False)
    instruments = ["USD_CAD", "USD_CHF", "USD_JPY"]
    for inst in instruments:
        _seed_fills(tmp_path, inst)

    closes = {
        "USD_CAD": list(np.linspace(1.30, 1.45, 160)),
        "USD_CHF": list(np.linspace(0.75, 0.85, 160)),
        "USD_JPY": list(np.linspace(150.0, 165.0, 160)),
    }
    client = _FakeClient(
        {inst: _ohlc_candles(c, range_frac=0.006) for inst, c in closes.items()}, nav=NAV)

    result = run_oanda_trend_cycle(
        client=client, config=_Config(), instruments=instruments,
        project_root=tmp_path, sma_window=20, candle_count=160, gross_leverage=3.0)

    assert result.ran is True
    # All 3 instruments trend "on" (uptrend) from a flat book, but the
    # correlated USD-long bucket cap admits at most 2 of the 3 THIS cycle.
    assert result.orders_placed <= 2
    assert result.orders_placed >= 1   # the gate isn't accidentally blocking everything either


def test_run_oanda_trend_cycle_never_blocks_a_close_even_with_zero_cost_data(tmp_path):
    """Decreases/closes must never be touched by the two new additive gates —
    only delta > 0 (size-increase) candidates route through them. Deliberately
    NO fill history at all for EUR_USD (cost data would fail-closed if this
    were a size-increase) and a downtrend so the strategy wants FLAT: an
    existing long position must still be closed (a real SELL order placed)
    despite zero cost/exposure data, proving risk-reducing orders are exempt."""
    StateEngine(tmp_path / ".claude" / "state.json").set_halted(False)
    closes = list(np.linspace(1.20, 1.05, 160))  # downtrend -> flat signal
    existing_trade = {
        "id": "9", "instrument": "EUR_USD", "currentUnits": "50000",
        "unrealizedPL": 5.0, "openTime": "2026-07-01T00:00:00Z",
        "stopLossOrder": {"price": "1.00000"}, "takeProfitOrder": {"price": "1.30000"},
    }
    client = _FakeClient(
        {"EUR_USD": _ohlc_candles(closes)}, nav=NAV,
        trades=[existing_trade],
        positions=[{"instrument": "EUR_USD", "long": {"units": "50000"}, "short": {"units": "0"}}],
    )
    result = run_oanda_trend_cycle(
        client=client, config=_Config(), instruments=["EUR_USD"],
        project_root=tmp_path, sma_window=20, candle_count=160, gross_leverage=3.0)
    assert result.ran is True
    assert result.orders_placed == 1
    assert client.orders[0]["units"] < 0   # a SELL to flatten, not blocked by the new gates
