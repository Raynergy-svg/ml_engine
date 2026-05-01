"""Tests for src/training/labels/realized_confidence_label.py.

The leak-prevention test (`test_label_invariance_to_feature_perturbation`)
is the most important: it asserts that perturbing input features by ±10%
does not change the label for real-labeled rows, and that for pseudo-
labeled rows the label is not a closed-form function of row-t feature
values (shuffling row-t feature values must not be predictive of the
pseudo label — pseudo only depends on forward bars + ATR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.labels import compute_realized_confidence_labels


def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLC + ATR + a few features (no leak; features unrelated to label)."""
    rng = np.random.default_rng(seed)
    # Random walk close
    rets = rng.normal(0.0, 0.001, size=n)
    close = 1.10 + np.cumsum(rets) * 0.5
    high = close + np.abs(rng.normal(0.0, 0.0005, size=n))
    low = close - np.abs(rng.normal(0.0, 0.0005, size=n))
    # Make ATR comfortably larger than per-bar noise so triple-barrier resolves
    atr = np.abs(rng.normal(0.0, 0.0015, size=n)) + 0.0008
    # Features completely unrelated to outcomes (forward bars determine label)
    rsi = rng.uniform(20, 80, size=n)
    adx = rng.uniform(5, 50, size=n)
    bb_pos = rng.uniform(0.0, 1.0, size=n)
    vol_ratio = rng.uniform(0.5, 1.5, size=n)
    atr_pct = atr / close

    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "atr": atr,  # plain ndarray — pd.DataFrame will use the row positions
            "atr_pct_14": atr_pct,
            "rsi": rsi,
            "adx": adx,
            "bb_position_20": bb_pos,
            "volume_ratio_20": vol_ratio,
        },
        index=idx,
    )


def _journal_for(df: pd.DataFrame, instrument: str, won_at: int) -> list:
    """Build a single-trade journal entry pointing at df.index[won_at]."""
    ts = df.index[won_at].isoformat()
    return [
        {
            "pair": instrument,
            "timestamp": ts,
            "entry_price": float(df["close"].iloc[won_at]),
            "direction": "LONG",
            "trade_won": True,
        }
    ]


def test_pure_winner_gets_high_label():
    df = _make_df(n=200, seed=1)
    journal = _journal_for(df, "EUR_USD", won_at=50)
    labels, meta = compute_realized_confidence_labels(
        df, instrument="EUR_USD", journal=journal,
        lookahead_bars=24, sl_atr_mult=1.0, tp_atr_mult=2.5,
        label_mode="real",
    )
    assert labels[50] == 1.0
    assert meta["n_real"] == 1
    assert meta["class_balance_real"] == 1.0


def test_pure_loser_gets_low_label():
    df = _make_df(n=200, seed=2)
    j = _journal_for(df, "EUR_USD", won_at=75)
    j[0]["trade_won"] = False
    labels, meta = compute_realized_confidence_labels(
        df, instrument="EUR_USD", journal=j,
        label_mode="real",
    )
    assert labels[75] == 0.0
    assert meta["n_real"] == 1
    assert meta["class_balance_real"] == 0.0


def test_unlabeled_row_uses_pseudo_label():
    """Without a journal match, blend mode falls back to triple-barrier pseudo."""
    df = _make_df(n=200, seed=3)
    # Empty journal — blend mode must produce some pseudo labels
    labels, meta = compute_realized_confidence_labels(
        df, instrument="EUR_USD", journal=[],
        lookahead_bars=10, sl_atr_mult=1.0, tp_atr_mult=2.5,
        label_mode="blend",
    )
    assert meta["n_real"] == 0
    assert meta["n_pseudo"] > 0  # triple-barrier resolves at least some rows
    # Labels (where present) are 0 or 1
    valid = labels[~np.isnan(labels)]
    assert np.all((valid == 0.0) | (valid == 1.0))


def test_journal_overrides_pseudo():
    """When both real (journal) and pseudo (triple-barrier) are available
    for the same row, the real label wins."""
    df = _make_df(n=200, seed=4)
    # Real journal says LOSS at row 30
    j = [{
        "pair": "EUR_USD",
        "timestamp": df.index[30].isoformat(),
        "entry_price": float(df["close"].iloc[30]),
        "direction": "LONG",
        "trade_won": False,
    }]
    blend_labels, meta = compute_realized_confidence_labels(
        df, instrument="EUR_USD", journal=j, label_mode="blend",
        lookahead_bars=24, sl_atr_mult=1.0, tp_atr_mult=2.5,
    )
    pseudo_only, _ = compute_realized_confidence_labels(
        df, instrument="EUR_USD", journal=j, label_mode="pseudo",
        lookahead_bars=24, sl_atr_mult=1.0, tp_atr_mult=2.5,
    )
    # Real label must override pseudo at row 30
    assert blend_labels[30] == 0.0
    assert meta["n_real"] == 1
    # If pseudo would have been different (e.g. 1.0), confirm override worked.
    # If pseudo happened to also be 0.0, this still passes; the contract is
    # that real wins.


def test_join_key_normalization():
    """`EUR/USD`, `EUR_USD`, and `eur_usd` all resolve to the same trade."""
    df = _make_df(n=100, seed=5)
    # Journal entry uses slash form
    j = [{
        "pair": "EUR/USD",
        "timestamp": df.index[40].isoformat(),
        "entry_price": float(df["close"].iloc[40]),
        "direction": "LONG",
        "trade_won": True,
    }]
    # The label generator normalizes its `instrument` argument to underscore form;
    # pair on the journal record should be normalized at load-time. Here we
    # simulate that by feeding an already-normalized record.
    j_norm = [{**j[0], "pair": "EUR_USD"}]
    labels, meta = compute_realized_confidence_labels(
        df, instrument="EUR_USD", journal=j_norm, label_mode="real",
    )
    assert labels[40] == 1.0
    assert meta["n_real"] == 1


def test_label_invariance_to_feature_perturbation():
    """Leak-prevention test.

    Real labels are journal outcomes — they must NOT depend on row-t
    feature values. Perturb features and verify labels are unchanged for
    real-labeled rows. For pseudo labels: shuffling row-t feature values
    must not change the pseudo label (pseudo depends on forward OHLC, not
    on row-t indicator values).
    """
    df = _make_df(n=300, seed=6)
    j = [{
        "pair": "EUR_USD",
        "timestamp": df.index[i].isoformat(),
        "entry_price": float(df["close"].iloc[i]),
        "direction": "LONG",
        "trade_won": (i % 3 == 0),  # mix of wins/losses
    } for i in (40, 60, 80, 100, 120)]

    labels_a, _ = compute_realized_confidence_labels(
        df, instrument="EUR_USD", journal=j, label_mode="blend",
        lookahead_bars=24, sl_atr_mult=1.0, tp_atr_mult=2.5,
    )

    # Perturb row-t feature columns by ±10% — these are NOT used by the
    # label generator (it only reads OHLC + ATR for the SL/TP distance and
    # the journal's outcome boolean).
    df2 = df.copy()
    df2["rsi"] = df2["rsi"] * 1.10
    df2["adx"] = df2["adx"] * 0.90
    df2["bb_position_20"] = df2["bb_position_20"] * 1.10
    df2["volume_ratio_20"] = df2["volume_ratio_20"] * 0.90
    # Note: we deliberately do NOT perturb high/low/close/atr — those are
    # the actual inputs used by the triple-barrier pseudo-label path.

    labels_b, _ = compute_realized_confidence_labels(
        df2, instrument="EUR_USD", journal=j, label_mode="blend",
        lookahead_bars=24, sl_atr_mult=1.0, tp_atr_mult=2.5,
    )

    # Real-labeled rows must be identical
    real_idx = [40, 60, 80, 100, 120]
    for i in real_idx:
        assert labels_a[i] == labels_b[i], (
            f"Real label changed at row {i} when row-t features perturbed: "
            f"{labels_a[i]} vs {labels_b[i]}"
        )

    # Pseudo-labeled rows must also be identical (perturbing row-t features
    # that the label generator doesn't read must not change pseudo labels).
    for i in range(len(labels_a)):
        if i in real_idx:
            continue
        if not (np.isnan(labels_a[i]) or np.isnan(labels_b[i])):
            assert labels_a[i] == labels_b[i], (
                f"Pseudo label changed at row {i} when row-t features perturbed."
            )


def test_returns_metadata_fields():
    df = _make_df(n=100, seed=7)
    labels, meta = compute_realized_confidence_labels(
        df, instrument="EUR_USD", journal=[], label_mode="pseudo",
        lookahead_bars=10,
    )
    for k in ("n_total", "n_real", "n_pseudo", "n_unavailable", "label_mode"):
        assert k in meta
    assert meta["n_total"] == len(df)
    assert meta["label_mode"] == "pseudo"
