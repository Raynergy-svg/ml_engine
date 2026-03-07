from __future__ import annotations

import pandas as pd

from scripts.export_pair_backtest_trades import (
    _apply_historical_label_overrides,
    _resolve_primary_probability,
    _simulate_trade_path,
)
from src.scanner.config import ScannerConfig
from src.scanner.results import PairAnalysis


def test_resolve_primary_probability_falls_back_to_confidence_proxy() -> None:
    analysis = PairAnalysis(
        pair="AUD_NZD",
        direction="SHORT",
        confidence=0.62,
        tcn_probability=0.5,
    )

    probability, source = _resolve_primary_probability(analysis, {})

    assert source == "confidence_proxy"
    assert round(probability, 4) == 0.81


def test_simulate_trade_path_hits_take_profit_for_short() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.2000, 1.1998, 1.1990, 1.1985],
            "high": [1.2002, 1.1999, 1.1992, 1.1986],
            "low": [1.1998, 1.1988, 1.1970, 1.1978],
            "close": [1.2000, 1.1990, 1.1975, 1.1980],
        },
        index=index,
    )

    result = _simulate_trade_path(
        df_raw=df,
        start_idx=0,
        direction="SHORT",
        entry_price=1.2000,
        pip_value=0.0001,
        sl_pips=15.0,
        tp_pips=10.0,
        max_hold_bars=3,
    )

    assert result["exit_reason"] == "take_profit"
    assert result["holding_bars"] == 1
    assert round(result["pnl_pips"], 4) == 10.0


def test_historical_label_overrides_apply_lenient_profile() -> None:
    cfg = ScannerConfig()

    _apply_historical_label_overrides(
        cfg,
        label_profile="lenient",
        weighted_vote_threshold=None,
        max_uncertainty_score=None,
        max_model_disagreement=None,
        max_spread_pips=None,
        min_liquidity_score=None,
    )

    assert cfg.sub_inference_tradeable_only is False
    assert cfg.weighted_vote_threshold == 0.48
    assert cfg.regime_vote_thresholds["EXTREME"] == 0.60
    assert cfg.max_uncertainty_score == 0.85
    assert cfg.max_spread_pips == 5.0
