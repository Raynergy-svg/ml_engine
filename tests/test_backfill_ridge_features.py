"""Tests for the live-journal ``ridge_features`` backfill utility.

Covers:
- 14-feature entry → backfilled to 24-feature vector with correct shape
- Idempotency: re-running backfill on a 24-feature entry leaves it unchanged
- Field preservation: every other key in the entry is preserved verbatim
- Skip behaviour when no OHLCV CSV exists for the trade's pair
- ``derive_ridge_features`` produces matching instrument one-hot
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Make scripts/ importable for the backfill module
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import backfill_ridge_features as bf  # type: ignore  # noqa: E402


# ─────────────────────────────────────────────────────────────────
# Fixtures: synthetic OHLCV CSV + saved model artifact
# ─────────────────────────────────────────────────────────────────

PAIR = "EUR_USD"
N_FEATURES = 24
FEATURE_NAMES = [
    "atr_pct_10", "atr_pct_20", "volatility_10", "volatility_20",
    "sma_ratio_20", "ema_ratio_26", "macd_norm", "rsi_norm",
    "zscore_20", "volume_ratio_10", "volume_ratio_20",
    "returns_5", "returns_10", "returns_20",
    # 10 instrument one-hots
    "instrument_EUR_USD", "instrument_GBP_USD", "instrument_USD_JPY",
    "instrument_USD_CHF", "instrument_AUD_USD", "instrument_USD_CAD",
    "instrument_NZD_USD", "instrument_EUR_GBP", "instrument_EUR_JPY",
    "instrument_GBP_JPY",
]


def _make_synthetic_csv(path: Path, pair: str, n_rows: int = 400) -> None:
    """Write a synthetic OHLCV CSV the backfill loader can ingest."""
    rng = np.random.default_rng(42)
    base = 1.10
    rets = rng.normal(0, 0.001, size=n_rows)
    close = base * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0005, size=n_rows)))
    low = close * (1 - np.abs(rng.normal(0, 0.0005, size=n_rows)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(100, 1000, size=n_rows).astype(float)
    times = pd.date_range("2026-04-01 00:00:00", periods=n_rows, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "time": times.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    df.to_csv(path, index=False)


def _make_synthetic_model(path: Path) -> None:
    """Write a minimal pickle the backfill module can introspect.

    We don't need a real estimator — only ``feature_names`` + ``n_features``
    are inspected. A trivial stub is sufficient for the schema discovery path.
    """
    payload = {
        "model": None,
        "scaler": None,
        "metrics": {},
        "config": {},
        "feature_names": list(FEATURE_NAMES),
        "n_features": N_FEATURES,
        "model_type": "lightgbm",
        "sklearn_version": "stub",
        "lightgbm_version": "stub",
        "saved_at": "2026-05-01T00:00:00",
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


@pytest.fixture
def env(tmp_path: Path):
    """Wire a fully isolated env: market_data dir, model file, journal file."""
    market_dir = tmp_path / "market_data"
    market_dir.mkdir()
    csv = market_dir / f"oanda_{PAIR}_H1_live_20260415_120000.csv"
    _make_synthetic_csv(csv, PAIR)

    model_path = tmp_path / "ridge_confidence.pkl"
    _make_synthetic_model(model_path)

    journal_path = tmp_path / "trade_journal_rl.json"

    # Reset cache between tests
    bf._OHLCV_CACHE.clear()  # type: ignore[attr-defined]

    return {
        "market_dir": market_dir,
        "model_path": model_path,
        "journal_path": journal_path,
    }


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

def test_14_feature_entry_backfilled_to_24(env):
    """14-length ridge_features should be replaced with 24-length vector."""
    journal = [{
        "trade_id": "t1",
        "pair": PAIR,
        "timestamp": "2026-04-15T10:00:00+00:00",
        "ridge_features": [0.0] * 14,
        "outcome": {"trade_won": True},
        "confidence": 60.0,
    }]
    env["journal_path"].write_text(json.dumps(journal))

    summary = bf.backfill_journal(
        journal_path=env["journal_path"],
        model_path=env["model_path"],
        market_dir=env["market_dir"],
    )

    assert summary.n_backfilled == 1
    out = json.loads(env["journal_path"].read_text())
    rf = out[0]["ridge_features"]
    assert len(rf) == N_FEATURES
    # Instrument one-hot for EUR_USD must be 1.0; others 0.0
    eur_idx = FEATURE_NAMES.index("instrument_EUR_USD")
    assert rf[eur_idx] == 1.0
    for name in (
        "instrument_GBP_USD", "instrument_USD_JPY", "instrument_USD_CHF",
        "instrument_AUD_USD", "instrument_USD_CAD", "instrument_NZD_USD",
        "instrument_EUR_GBP", "instrument_EUR_JPY", "instrument_GBP_JPY",
    ):
        assert rf[FEATURE_NAMES.index(name)] == 0.0


def test_none_entry_backfilled(env):
    """``ridge_features=None`` (the live-journal default) is treated as
    "needs backfill"."""
    journal = [{
        "trade_id": "t1",
        "pair": PAIR,
        "timestamp": "2026-04-15T10:00:00+00:00",
        "ridge_features": None,
        "outcome": {"trade_won": False},
    }]
    env["journal_path"].write_text(json.dumps(journal))

    summary = bf.backfill_journal(
        journal_path=env["journal_path"],
        model_path=env["model_path"],
        market_dir=env["market_dir"],
    )

    assert summary.n_backfilled == 1
    out = json.loads(env["journal_path"].read_text())
    assert len(out[0]["ridge_features"]) == N_FEATURES


def test_idempotent(env):
    """Running backfill twice yields no further changes."""
    journal = [{
        "trade_id": "t1",
        "pair": PAIR,
        "timestamp": "2026-04-15T10:00:00+00:00",
        "ridge_features": None,
    }]
    env["journal_path"].write_text(json.dumps(journal))

    s1 = bf.backfill_journal(env["journal_path"], env["model_path"], env["market_dir"])
    after_first = json.loads(env["journal_path"].read_text())
    assert s1.n_backfilled == 1
    assert len(after_first[0]["ridge_features"]) == N_FEATURES

    # Reset cache so the second pass re-loads the CSV (still deterministic)
    bf._OHLCV_CACHE.clear()  # type: ignore[attr-defined]
    s2 = bf.backfill_journal(env["journal_path"], env["model_path"], env["market_dir"])
    after_second = json.loads(env["journal_path"].read_text())
    assert s2.n_backfilled == 0
    assert s2.n_already_24 == 1
    assert after_first == after_second


def test_field_preservation(env):
    """Backfill must preserve every other field on the entry."""
    journal = [{
        "trade_id": "preserved-id",
        "pair": PAIR,
        "timestamp": "2026-04-15T10:00:00+00:00",
        "ridge_features": None,
        "direction": "LONG",
        "confidence": 73.5,
        "outcome": {"trade_won": True, "exit_price": 1.105, "pnl": 12.5},
        "agents": {"trend": {"score": 0.7}},
        "extra_nested": {"a": [1, 2, 3], "b": {"c": "d"}},
    }]
    env["journal_path"].write_text(json.dumps(journal))

    bf.backfill_journal(env["journal_path"], env["model_path"], env["market_dir"])

    out = json.loads(env["journal_path"].read_text())[0]
    # ridge_features changed, but every other key must be byte-identical
    assert out["trade_id"] == "preserved-id"
    assert out["pair"] == PAIR
    assert out["timestamp"] == "2026-04-15T10:00:00+00:00"
    assert out["direction"] == "LONG"
    assert out["confidence"] == 73.5
    assert out["outcome"] == {"trade_won": True, "exit_price": 1.105, "pnl": 12.5}
    assert out["agents"] == {"trend": {"score": 0.7}}
    assert out["extra_nested"] == {"a": [1, 2, 3], "b": {"c": "d"}}


def test_skip_when_no_csv_for_pair(env):
    """If we lack OHLCV for a pair, the entry is skipped + recorded — not crashed."""
    journal = [
        {
            "trade_id": "ok",
            "pair": PAIR,
            "timestamp": "2026-04-15T10:00:00+00:00",
            "ridge_features": None,
        },
        {
            "trade_id": "no_data",
            "pair": "EXOTIC_PAIR",  # no CSV under env["market_dir"]
            "timestamp": "2026-04-15T10:00:00+00:00",
            "ridge_features": None,
        },
    ]
    env["journal_path"].write_text(json.dumps(journal))

    summary = bf.backfill_journal(env["journal_path"], env["model_path"], env["market_dir"])
    assert summary.n_backfilled == 1
    assert summary.n_skipped_no_csv == 1

    out = json.loads(env["journal_path"].read_text())
    assert len(out[0]["ridge_features"]) == N_FEATURES   # backfilled
    assert out[1]["ridge_features"] is None              # untouched


def test_skip_when_timestamp_out_of_band(env):
    """Trade timestamp far before any OHLCV → skip with timestamp_oob.

    Use a tight ``max_bar_age_days`` to force the skip path on a synthetic
    CSV whose first row is in 2026.
    """
    journal = [{
        "trade_id": "old",
        "pair": PAIR,
        "timestamp": "1999-01-01T00:00:00+00:00",  # way before synthetic CSV start
        "ridge_features": None,
    }]
    env["journal_path"].write_text(json.dumps(journal))

    summary = bf.backfill_journal(
        env["journal_path"], env["model_path"], env["market_dir"],
        max_bar_age_days=30.0,
    )
    assert summary.n_backfilled == 0
    assert summary.n_skipped_timestamp_oob == 1


def test_derive_ridge_features_returns_correct_length(env):
    """Direct unit test of the per-trade helper."""
    feature_names, _ = bf.load_model_feature_names(env["model_path"])
    vec, reason, bar_age = bf.derive_ridge_features(
        pair=PAIR,
        timestamp="2026-04-15T10:00:00+00:00",
        feature_names=feature_names,
        market_dir=env["market_dir"],
    )
    assert reason is None
    assert len(vec) == N_FEATURES
    assert bar_age is not None and bar_age >= 0.0
    # Instrument one-hot must select PAIR
    eur_idx = feature_names.index("instrument_EUR_USD")
    assert vec[eur_idx] == 1.0


def test_dry_run_does_not_write(env):
    """``dry_run=True`` must not modify the journal on disk."""
    journal = [{
        "trade_id": "t1",
        "pair": PAIR,
        "timestamp": "2026-04-15T10:00:00+00:00",
        "ridge_features": None,
    }]
    env["journal_path"].write_text(json.dumps(journal))
    pre = env["journal_path"].read_text()

    summary = bf.backfill_journal(
        env["journal_path"], env["model_path"], env["market_dir"], dry_run=True,
    )
    assert summary.n_backfilled == 1
    post = env["journal_path"].read_text()
    assert pre == post  # disk unchanged
