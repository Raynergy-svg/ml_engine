"""End-to-end test: synthetic df + journal → load_ridge_data returns
a labeled training set with no NaN labels and the expected real/pseudo split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.modular_data_loaders import load_ridge_data


def _make_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.001, size=n)
    close = 1.10 + np.cumsum(rets) * 0.5
    high = close + np.abs(rng.normal(0.0, 0.0005, size=n))
    low = close - np.abs(rng.normal(0.0, 0.0005, size=n))
    volume = rng.uniform(1000, 5000, size=n).astype(float)
    atr_pct = np.abs(rng.normal(0.0, 0.0008, size=n)) + 0.0005
    atr = atr_pct * close

    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr": atr,
            "atr_pct_14": atr_pct,
            "atr_pct_10": atr_pct,
            "atr_pct_20": atr_pct,
            "rsi": rng.uniform(20, 80, size=n),
            "rsi_norm": rng.uniform(0, 1, size=n),
            "adx": rng.uniform(5, 50, size=n),
            "bb_width_20": rng.uniform(0.001, 0.005, size=n),
            "bb_position_20": rng.uniform(0.0, 1.0, size=n),
            "volatility_5": np.abs(rng.normal(0.0, 0.0005, size=n)),
            "volatility_10": np.abs(rng.normal(0.0, 0.0005, size=n)),
            "volatility_20": np.abs(rng.normal(0.0, 0.0005, size=n)),
            "volume_ratio_10": rng.uniform(0.5, 1.5, size=n),
            "volume_ratio_20": rng.uniform(0.5, 1.5, size=n),
            "returns": rng.normal(0.0, 0.001, size=n),
            "returns_1": rng.normal(0.0, 0.001, size=n),
        },
        index=idx,
    )
    return df


def test_load_ridge_data_returns_no_nan_labels(tmp_path: Path):
    df = _make_df(n=400, seed=11)
    # Empty journal so triple-barrier pseudo path is the only label source
    j_path = tmp_path / "empty.json"
    j_path.write_text("[]")

    result = load_ridge_data(
        df,
        instrument="EUR_USD",
        journal_path=str(j_path),
        label_mode="pseudo",
        lookahead_bars=10,
        confidence_window=5,
    )

    # No NaN survives
    assert not np.any(np.isnan(result["y_train"]))
    assert not np.any(np.isnan(result["y_val"]))
    # Labels are in [20, 95] (rescaled binary)
    assert float(result["y_train"].min()) >= 20.0
    assert float(result["y_train"].max()) <= 95.0
    # Labels are binary — only two unique values (20.0 and 95.0)
    unique_y = np.unique(result["y_train"])
    assert len(unique_y) <= 2
    for v in unique_y:
        assert v in (20.0, 95.0), f"unexpected y value: {v}"


def test_load_ridge_data_records_label_metadata(tmp_path: Path):
    df = _make_df(n=300, seed=12)
    j_path = tmp_path / "j.json"
    # Single real win
    journal = [{
        "pair": "EUR_USD",
        "timestamp": df.index[100].isoformat(),
        "entry_price": float(df["close"].iloc[100]),
        "direction": "LONG",
        "outcome": {"trade_won": True, "exit_reason": "tp_hit"},
    }]
    j_path.write_text(json.dumps(journal))

    result = load_ridge_data(
        df,
        instrument="EUR_USD",
        journal_path=str(j_path),
        label_mode="blend",
        lookahead_bars=10,
        confidence_window=5,
    )

    meta = result["label_metadata"]
    assert meta["n_real"] == 1
    assert meta["n_pseudo"] > 0
    assert meta["score_range"] == [20.0, 95.0]
    assert meta["leak_fix_version"] == "2026-04-30"


def test_load_ridge_data_signature_backward_compat():
    """Existing callers pass no extra args — must still work."""
    df = _make_df(n=200, seed=13)
    # No instrument, no journal_path, no label_mode — defaults must produce
    # a usable result (pseudo-labels via triple-barrier).
    result = load_ridge_data(df)
    assert "X_train" in result
    assert "y_train" in result
    assert "label_metadata" in result
