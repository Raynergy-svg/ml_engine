"""Embargo (purged temporal split) tests for modular data loaders.

Audit finding L-C1 (docs/training-infra-audit-2026-06-10.md): forward-looking
labels leak across the train/val boundary when the temporal split has no gap.
These tests assert that each fixed loader applies an embargo gap >= its own
label lookahead/horizon BY DEFAULT, and that gap=0 remains reachable by
passing it explicitly (A/B comparison escape hatch).

NO MOCKS: real synthetic OHLCV DataFrames, real loader calls, real label
generators, real RobustScaler fits.

Tracer technique: the loader's expected feature columns are set to the row
index (0..n-1). Loaders that don't scale X (load_rf_data, load_ridge_data,
load_volatility_regime_data) expose the tracer directly; loaders that scale
(load_tcn_data via load_direction_data, load_forward_volatility_data) use an
invertible RobustScaler, so the original row index of any train/val/test
row is recovered exactly via scaler.inverse_transform. The index distance
between the last train row/sequence and the first val row/sequence is then
asserted directly against the loader's horizon.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.core.modular_data_loaders import (
    load_forward_volatility_data,
    load_rf_data,
    load_ridge_data,
    load_tcn_data,
    load_volatility_regime_data,
    temporal_split,
)

# ---------------------------------------------------------------------------
# Synthetic data builders (real frames, no mocks)
# ---------------------------------------------------------------------------

N = 1200  # bars per synthetic frame — big enough for 0.7/0.2/0.1 + gaps


def _ohlcv(n: int = N, seed: int = 7) -> pd.DataFrame:
    """Random-walk OHLCV frame with a real time column."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.002, n)
    close = 1.10 + np.cumsum(steps)
    close = np.maximum(close, 0.5)  # keep prices positive
    spread = np.abs(rng.normal(0.0, 0.0008, n)) + 0.0004
    high = close + spread
    low = close - spread
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(500, 5000, n).astype(np.float64)
    return pd.DataFrame({
        'time': pd.date_range('2025-01-01', periods=n, freq='15min', tz='UTC'),
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })


def _tracer_frame(feature_names, n: int = N, seed: int = 7) -> pd.DataFrame:
    """OHLCV frame whose loader-expected feature columns equal the row index.

    'returns_1' is always present so the loaders skip
    compute_normalized_features() and our tracer columns survive intact.
    """
    df = _ohlcv(n, seed)
    tracer = np.arange(n, dtype=np.float64)
    for name in feature_names:
        df[name] = tracer
    if 'returns_1' not in df.columns:
        df['returns_1'] = tracer
    return df


# Feature lists mirroring each loader's preferred-feature names
# (see get_normalized_feature_names()['risk'/'confidence'] and the inline
# lists in load_volatility_regime_data / load_forward_volatility_data).
RISK_FEATURES = [
    'atr_pct_5', 'atr_pct_10', 'atr_pct_14', 'atr_pct_20',
    'volatility_5', 'volatility_10', 'volatility_20', 'tr_pct',
    'hl_range_pct', 'upper_wick_ratio', 'lower_wick_ratio',
    'zscore_10', 'zscore_20', 'returns_1', 'returns_5', 'returns_10',
    'volume_ratio_10', 'volume_zscore',
]
CONFIDENCE_FEATURES = [
    'atr_pct_10', 'atr_pct_20', 'volatility_10', 'volatility_20',
    'sma_ratio_20', 'ema_ratio_26', 'macd_norm', 'rsi_norm', 'zscore_20',
    'volume_ratio_10', 'volume_ratio_20', 'returns_5', 'returns_10',
    'returns_20',
]
VOL_REGIME_FEATURES = [
    'atr_pct_5', 'atr_pct_10', 'atr_pct_14', 'atr_pct_20',
    'volatility_5', 'volatility_10', 'volatility_20',
    'bb_width_norm', 'bb_width_20', 'high_low_range',
    'returns_1', 'returns_5', 'returns_10',
    'volume_ratio_10', 'volume_ratio_20', 'adx',
]
FORWARD_VOL_FEATURES = [
    'atr_pct_5', 'atr_pct_10', 'atr_pct_20',
    'volatility_5', 'volatility_10', 'volatility_20',
    'tr_pct', 'hl_range_pct',
    'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
    'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
    'returns_1', 'returns_2', 'returns_5', 'returns_10',
    'hour_sin', 'hour_cos',
]


def _boundary_distances(result, col: int = 0):
    """Tracer distance (in original bar indices) across both boundaries."""
    train_val = float(result['X_val'][0, col]) - float(result['X_train'][-1, col])
    val_test = float(result['X_test'][0, col]) - float(result['X_val'][-1, col])
    return train_val, val_test


# ---------------------------------------------------------------------------
# temporal_split arithmetic (foundation all loaders rely on)
# ---------------------------------------------------------------------------

class TestTemporalSplitGap:
    def test_gap_carved_from_val_and_test_heads(self):
        train_idx, val_idx, test_idx = temporal_split(1000, 0.7, 0.2, 0.1, gap=24)
        assert train_idx[-1] == 699
        assert val_idx[0] - train_idx[-1] - 1 == 24
        assert test_idx[0] - val_idx[-1] - 1 == 24
        # Train untouched; gap shrinks val/test only
        assert len(train_idx) == 700

    def test_gap_zero_is_contiguous(self):
        train_idx, val_idx, test_idx = temporal_split(1000, 0.7, 0.2, 0.1, gap=0)
        assert val_idx[0] - train_idx[-1] == 1
        assert test_idx[0] - val_idx[-1] == 1


# ---------------------------------------------------------------------------
# Fix 1: load_volatility_regime_data — default gap = vol_horizon_bars (24)
# ---------------------------------------------------------------------------

class TestVolatilityRegimeEmbargo:
    HORIZON = 24  # vol_horizon_bars hardcoded in the loader

    def test_default_gap_is_label_horizon(self):
        df = _tracer_frame(VOL_REGIME_FEATURES)
        res = load_volatility_regime_data(df)
        assert res['embargo_gap'] == self.HORIZON
        train_val, val_test = _boundary_distances(res)
        # Tracer rows are contiguous (head warmup trim + tail sentinel drop
        # only), so the boundary distance is exactly gap+1 bars.
        assert train_val == self.HORIZON + 1
        assert val_test == self.HORIZON + 1
        assert train_val - 1 >= self.HORIZON  # the audit criterion

    def test_gap_zero_explicit_override(self):
        df = _tracer_frame(VOL_REGIME_FEATURES)
        res = load_volatility_regime_data(df, gap=0)
        assert res['embargo_gap'] == 0
        train_val, val_test = _boundary_distances(res)
        assert train_val == 1
        assert val_test == 1

    def test_train_set_unchanged_by_embargo(self):
        df = _tracer_frame(VOL_REGIME_FEATURES)
        res_default = load_volatility_regime_data(df)
        res_gap0 = load_volatility_regime_data(df, gap=0)
        np.testing.assert_array_equal(res_default['X_train'], res_gap0['X_train'])
        # Embargoed val is exactly the gap0 val minus its first HORIZON rows
        np.testing.assert_array_equal(
            res_default['X_val'], res_gap0['X_val'][self.HORIZON:]
        )


# ---------------------------------------------------------------------------
# Fix 2: load_tcn_data — gap forwarded to load_direction_data (= lookahead)
# ---------------------------------------------------------------------------

def _tcn_frame(n: int = N, seed: int = 7) -> pd.DataFrame:
    """OHLCV + tracer + noise features; returns_1 present to skip recompute."""
    rng = np.random.default_rng(seed)
    df = _ohlcv(n, seed)
    df['returns_1'] = df['close'].pct_change().fillna(0.0)
    df['tracer_idx'] = np.arange(n, dtype=np.float64)
    for k in range(5):
        df[f'noise_{k}'] = rng.normal(0.0, 1.0, n)
    return df


def _tcn_boundary_distances(res):
    """Recover original row indices via the loader's fitted RobustScaler."""
    tracer_pos = res['feature_names'].index('tracer_idx')
    inv = res['scaler'].inverse_transform
    last_train = inv(res['X_train'][-1:].astype(np.float64))[0][tracer_pos]
    first_val = inv(res['X_val'][:1].astype(np.float64))[0][tracer_pos]
    last_val = inv(res['X_val'][-1:].astype(np.float64))[0][tracer_pos]
    first_test = inv(res['X_test'][:1].astype(np.float64))[0][tracer_pos]
    return round(first_val - last_train), round(first_test - last_val)


class TestTcnEmbargo:
    LOOKAHEAD = 6  # load_tcn_data default

    def test_default_gap_is_lookahead(self):
        res = load_tcn_data(_tcn_frame(), threshold=0.0)
        train_val, val_test = _tcn_boundary_distances(res)
        assert train_val == self.LOOKAHEAD + 1
        assert val_test == self.LOOKAHEAD + 1
        assert train_val - 1 >= self.LOOKAHEAD

    def test_gap_zero_explicit_override(self):
        res = load_tcn_data(_tcn_frame(), threshold=0.0)
        res0 = load_tcn_data(_tcn_frame(), threshold=0.0, gap=0)
        train_val0, val_test0 = _tcn_boundary_distances(res0)
        assert train_val0 == 1
        assert val_test0 == 1
        # gap carves from val/test heads only
        assert len(res['X_val']) == len(res0['X_val']) - self.LOOKAHEAD
        assert len(res['X_test']) == len(res0['X_test']) - self.LOOKAHEAD
        np.testing.assert_array_equal(res['X_train'], res0['X_train'])


# ---------------------------------------------------------------------------
# Fix 3: load_forward_volatility_data — sequence-level purge (= lookahead)
# ---------------------------------------------------------------------------

def _fv_boundary_distances(res):
    """Recover the bar index at each sequence's label position (last
    timestep) via the loader's fitted RobustScaler."""
    inv = res['scaler'].inverse_transform
    col = 0  # all tracer columns identical; column 0 suffices
    last_train = inv(res['X_train'][-1, -1:, :].astype(np.float64))[0][col]
    first_val = inv(res['X_val'][0, -1:, :].astype(np.float64))[0][col]
    last_val = inv(res['X_val'][-1, -1:, :].astype(np.float64))[0][col]
    first_test = inv(res['X_test'][0, -1:, :].astype(np.float64))[0][col]
    return round(first_val - last_train), round(first_test - last_val)


class TestForwardVolatilityEmbargo:
    LOOKAHEAD = 48  # loader default

    def test_default_gap_is_lookahead_sequences(self):
        df = _tracer_frame(FORWARD_VOL_FEATURES, n=1500)
        res = load_forward_volatility_data(df)
        assert res['embargo_gap'] == self.LOOKAHEAD
        train_val, val_test = _fv_boundary_distances(res)
        # Sequences are stride-1, so dropping `gap` sequences from the head
        # of val/test puts gap+1 bars between adjacent label positions.
        assert train_val == self.LOOKAHEAD + 1
        assert val_test == self.LOOKAHEAD + 1
        assert train_val - 1 >= self.LOOKAHEAD

    def test_gap_zero_explicit_override_preserves_contract(self):
        df = _tracer_frame(FORWARD_VOL_FEATURES, n=1500)
        res = load_forward_volatility_data(df)
        res0 = load_forward_volatility_data(df, gap=0)
        train_val0, val_test0 = _fv_boundary_distances(res0)
        assert train_val0 == 1
        assert val_test0 == 1
        # Same return contract, just fewer boundary rows in val/test
        assert set(res.keys()) == set(res0.keys())
        np.testing.assert_array_equal(res['X_train'], res0['X_train'])
        np.testing.assert_array_equal(res['X_val'], res0['X_val'][self.LOOKAHEAD:])
        np.testing.assert_array_equal(res['X_test'], res0['X_test'][self.LOOKAHEAD:])
        np.testing.assert_array_equal(res['y_val_reg'], res0['y_val_reg'][self.LOOKAHEAD:])
        np.testing.assert_array_equal(res['w_val'], res0['w_val'][self.LOOKAHEAD:])


# ---------------------------------------------------------------------------
# Fix 4a: load_rf_data — default gap = drawdown_horizon (10)
# ---------------------------------------------------------------------------

class TestRfEmbargo:
    HORIZON = 10  # drawdown_horizon default

    def test_default_gap_is_drawdown_horizon(self):
        df = _tracer_frame(RISK_FEATURES)
        res = load_rf_data(df)
        train_val, val_test = _boundary_distances(res)
        # RF rows are contiguous (head warmup 20 / tail horizon trim only)
        assert train_val == self.HORIZON + 1
        assert val_test == self.HORIZON + 1
        assert train_val - 1 >= self.HORIZON

    def test_gap_zero_explicit_override(self):
        df = _tracer_frame(RISK_FEATURES)
        res = load_rf_data(df)
        res0 = load_rf_data(df, gap=0)
        train_val0, val_test0 = _boundary_distances(res0)
        assert train_val0 == 1
        assert val_test0 == 1
        np.testing.assert_array_equal(res['X_train'], res0['X_train'])
        np.testing.assert_array_equal(res['X_val'], res0['X_val'][self.HORIZON:])


# ---------------------------------------------------------------------------
# Fix 4b: load_ridge_data — default gap = lookahead_bars (24)
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_journal(tmp_path):
    """Real empty journal file on real disk — forces pseudo-label-only path."""
    p = tmp_path / 'journal.json'
    p.write_text(json.dumps([]))
    return str(p)


def _ridge_frame(n: int = N, seed: int = 7) -> pd.DataFrame:
    df = _tracer_frame(CONFIDENCE_FEATURES, n=n, seed=seed)
    # ATR column required by the triple-barrier pseudo-label generator.
    # ~0.002 on a 0.002-sigma walk → TP/SL resolvable within 24 bars.
    df['atr'] = 0.002
    return df


class TestRidgeEmbargo:
    LOOKAHEAD = 24  # lookahead_bars default (triple-barrier walk)

    def test_default_gap_is_lookahead_bars(self, empty_journal):
        df = _ridge_frame()
        res = load_ridge_data(df, journal_path=empty_journal)
        train_val, val_test = _boundary_distances(res)
        # NaN-label rows may be dropped mid-series (triple-barrier timeouts),
        # so the original-bar distance is >= gap+1, not exactly gap+1.
        assert train_val >= self.LOOKAHEAD + 1
        assert val_test >= self.LOOKAHEAD + 1

    def test_gap_zero_explicit_override(self, empty_journal):
        df = _ridge_frame()
        res = load_ridge_data(df, journal_path=empty_journal)
        res0 = load_ridge_data(df, journal_path=empty_journal, gap=0)
        # Label generation is deterministic given the same df + journal, so
        # the embargoed splits are exact head-shifts of the gap=0 splits.
        np.testing.assert_array_equal(res['X_train'], res0['X_train'])
        np.testing.assert_array_equal(res['X_val'], res0['X_val'][self.LOOKAHEAD:])
        np.testing.assert_array_equal(res['y_val'], res0['y_val'][self.LOOKAHEAD:])
        assert len(res['X_val']) == len(res0['X_val']) - self.LOOKAHEAD
        assert len(res['X_test']) == len(res0['X_test']) - self.LOOKAHEAD
