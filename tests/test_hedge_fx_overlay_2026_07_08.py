"""No-mock tests for FX hedge-overlay forward marking in
src/hedge/hedged_shadow_lane.py — real tmp_path {PAIR}_D.csv panels, real
bucket maps. Proves an FX hedge leg's overlay P&L resolves (direct + inverse
pair), degrades honestly (no panel / no forward bar / bad tick), and that the
overlay stays ALL-OR-NOTHING. No unittest.mock anywhere."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.hedge.hedged_shadow_lane import (
    _is_fx_instrument,
    fx_instrument_forward_return,
    hedge_overlay_pnl,
)


def _write_fx_csv(panel_dir: Path, pair: str, closes) -> None:
    """closes: list of (date_str, close). open/high/low/volume are filler."""
    panel_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"date": d, "open": c, "high": c, "low": c, "close": c, "volume": 1000.0}
            for d, c in closes]
    pd.DataFrame(rows).to_csv(panel_dir / f"{pair}_D.csv", index=False)


@pytest.fixture
def panel_dir(tmp_path):
    d = tmp_path / "factor"
    # GBP_USD rises 1.25 -> 1.26 across the forward bar; CHF_JPY falls.
    _write_fx_csv(d, "GBP_USD", [("2026-05-01", 1.25), ("2026-05-02", 1.26)])
    _write_fx_csv(d, "CHF_JPY", [("2026-05-01", 170.0), ("2026-05-02", 168.3)])
    return d


def test_direct_pair_forward_return(panel_dir):
    ret, notes = fx_instrument_forward_return("GBP_USD", "2026-05-01", panel_dir)
    assert notes == []
    assert ret == pytest.approx(1.26 / 1.25 - 1.0)


def test_inverse_pair_forward_return(panel_dir):
    # JPY_CHF not cached -> inverse of CHF_JPY, exactly inverted.
    direct, _ = fx_instrument_forward_return("CHF_JPY", "2026-05-01", panel_dir)
    inv, notes = fx_instrument_forward_return("JPY_CHF", "2026-05-01", panel_dir)
    assert notes == []
    assert inv == pytest.approx(170.0 / 168.3 - 1.0)
    # inverse == -direct/(1+direct)
    assert inv == pytest.approx(-direct / (1.0 + direct))


def test_no_forward_bar_after_asof(panel_dir):
    ret, notes = fx_instrument_forward_return("GBP_USD", "2099-01-01", panel_dir)
    assert ret is None and notes == ["no_forward_bar:GBP_USD"]


def test_uncached_pair_either_direction(panel_dir):
    ret, notes = fx_instrument_forward_return("AUD_NZD", "2026-05-01", panel_dir)
    assert ret is None and notes == ["no_fx_panel:AUD_NZD"]


def test_implausible_return_dropped_as_bad_tick(panel_dir):
    _write_fx_csv(panel_dir, "EUR_USD", [("2026-05-01", 1.0), ("2026-05-02", 5.0)])  # +400%
    ret, notes = fx_instrument_forward_return("EUR_USD", "2026-05-01", panel_dir)
    assert ret is None and notes == ["implausible_return:EUR_USD"]


def test_is_fx_instrument_discriminator():
    from src.hedge.exposure_tags import CURRENCY_BUCKET_MAP_PATH, load_bucket_map
    ccy_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
    assert _is_fx_instrument("GBP_USD", ccy_map) is True
    assert _is_fx_instrument("SPY", ccy_map) is False        # equity market hedge
    assert _is_fx_instrument("XLK", ccy_map) is False        # equity sector hedge
    assert _is_fx_instrument("GBP_USD", None) is False       # no map -> not FX


def test_overlay_resolves_for_fx_leg(panel_dir):
    from src.hedge.exposure_tags import CURRENCY_BUCKET_MAP_PATH, load_bucket_map
    ccy_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
    applied = {"style": "direct_offset", "cost_known": False, "cost_estimate_dollars": None,
               "legs": [{"instrument": "GBP_USD", "direction": "short", "risk_home": 10000.0}]}
    overlay = hedge_overlay_pnl(applied, 100000.0, "2026-05-01", pd.DataFrame(), {},
                                currency_bucket_map=ccy_map, fx_panel_dir=panel_dir)
    gbp_ret = 1.26 / 1.25 - 1.0
    # short leg: sign -1 * risk_home * ret
    assert overlay["gross_pnl_dollars"] == pytest.approx(-1.0 * 10000.0 * gbp_ret)
    assert overlay["net_pnl_dollars"] is None  # cost unknown -> honest


def test_overlay_all_or_nothing_on_unresolved_fx_leg(panel_dir):
    from src.hedge.exposure_tags import CURRENCY_BUCKET_MAP_PATH, load_bucket_map
    ccy_map = load_bucket_map(CURRENCY_BUCKET_MAP_PATH)
    applied = {"style": "relative_value", "cost_known": False, "cost_estimate_dollars": None,
               "legs": [
                   {"instrument": "GBP_USD", "direction": "long", "risk_home": 5000.0},
                   {"instrument": "AUD_NZD", "direction": "short", "risk_home": 5000.0},  # uncached
               ]}
    overlay = hedge_overlay_pnl(applied, 100000.0, "2026-05-01", pd.DataFrame(), {},
                                currency_bucket_map=ccy_map, fx_panel_dir=panel_dir)
    # one leg unresolved -> WHOLE overlay None (never a silent partial)
    assert overlay["gross_pnl_dollars"] is None
    assert any("overlay_unresolved" in n for n in overlay["notes"])
