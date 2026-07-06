"""Backfill 24-feature ``ridge_features`` for live journal trades.

Context (2026-05-01 follow-up to the 2026-04-30 leak fix):
    Live trade journal entries store ``ridge_features`` (a 14- or
    None-length vector) that was suitable for the OLD 14-feature
    confidence model. The leak-fix retrained the joint confidence
    head with a NEW 24-feature schema (14 numeric features + 10
    instrument one-hots — see ``trained_data/models/joint/ridge_confidence.pkl``
    ``feature_names``). The mismatch prevented those live trades from
    contributing to calibrator refit.

This script:
1. Reads ``trained_data/trade_journal_rl.json``.
2. Identifies entries where ``len(ridge_features) != 24`` (None counts).
3. For each such entry, re-derives the 24-feature vector by:
     a. Locating an H1 OHLCV CSV for the trade's pair (newest under
        ``market_data/oanda_{PAIR}_H1_live_*.csv``).
     b. Computing the same indicator family the live scanner uses
        (``compute_normalized_features`` + ATR/RSI/ADX/BB extras
        from ``scripts/retrain_confidence_leak_fix._add_indicators``).
     c. Slicing the row at-or-just-before the trade's ``timestamp``.
     d. Aligning to the joint model's saved ``feature_names`` via
        ``align_features_to_model`` (same builder gates.py uses for
        14 numeric slots) and filling instrument one-hots manually
        per ``gates.py:1263-1266`` semantics.
4. Writes the updated journal back atomically (tmp + rename) and
   verifies the result via sha256.

Idempotent: re-running on an already-backfilled journal touches nothing.

Skip conditions (logged + recorded in summary, NOT a failure):
- No H1 CSV available for the pair (e.g. exotic pair).
- CSV does not contain the trade's timestamp range.
- Trade's ``ridge_features`` is already a 24-length vector.

Per the controller brief, this is the storage format going forward —
``src/scanner/engine.py`` and ``src/scanner/gates.py`` are NOT modified;
new live trades will already store a 24-feature vector through
``modular_inference.py:_extract_ridge_features`` once the new model
is in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.modular_data_loaders import (  # noqa: E402
    align_features_to_model,
    compute_normalized_features,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backfill_ridge_features")

# Defaults — overridable via CLI
_DEFAULT_JOURNAL = _PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"
_DEFAULT_MODEL = _PROJECT_ROOT / "trained_data" / "models" / "joint" / "ridge_confidence.pkl"
_DEFAULT_MARKET_DIR = _PROJECT_ROOT / "market_data"


@dataclass
class BackfillSummary:
    n_total: int = 0
    n_already_24: int = 0
    n_backfilled: int = 0
    n_skipped_no_csv: int = 0
    n_skipped_timestamp_oob: int = 0
    n_skipped_other: int = 0
    skip_details: List[Dict[str, Any]] = field(default_factory=list)
    pairs_touched: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_already_24": self.n_already_24,
            "n_backfilled": self.n_backfilled,
            "n_skipped_no_csv": self.n_skipped_no_csv,
            "n_skipped_timestamp_oob": self.n_skipped_timestamp_oob,
            "n_skipped_other": self.n_skipped_other,
            "skip_details": self.skip_details,
            "pairs_touched": sorted(set(self.pairs_touched)),
        }


# ─────────────────────────────────────────────────────────────────
# Indicator computation — mirrors scripts/retrain_confidence_leak_fix.py
# `_add_indicators` so the FEATURE BUILDER is shared. We import that
# helper to keep the schema in lockstep.
# ─────────────────────────────────────────────────────────────────

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Re-export of the retrain script's indicator builder.

    We re-implement here (instead of importing from scripts/) to avoid
    `sys.path` gymnastics in tests. Keeping the body identical to
    ``scripts/retrain_confidence_leak_fix._add_indicators`` is a
    maintenance contract; if you change one, change both.
    """
    out = compute_normalized_features(df).copy()
    high = out["high"].values
    low = out["low"].values
    close = out["close"].values
    volume = out["volume"].values

    # ATR(14)
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - np.concatenate([[close[0]], close[:-1]])),
        np.abs(low - np.concatenate([[close[0]], close[:-1]])),
    ])
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    out["atr"] = atr
    out["atr_pct_14"] = atr / np.maximum(close, 1e-10)

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1 / 14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1 / 14, adjust=False).mean().values
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    out["rsi"] = rsi
    out["rsi_norm"] = rsi / 100.0

    # ADX(14)
    plus_dm = np.where(
        (high - np.concatenate([[high[0]], high[:-1]])) >
        (np.concatenate([[low[0]], low[:-1]]) - low),
        np.maximum(high - np.concatenate([[high[0]], high[:-1]]), 0.0), 0.0,
    )
    minus_dm = np.where(
        (np.concatenate([[low[0]], low[:-1]]) - low) >
        (high - np.concatenate([[high[0]], high[:-1]])),
        np.maximum(np.concatenate([[low[0]], low[:-1]]) - low, 0.0), 0.0,
    )
    atr_safe = np.maximum(atr, 1e-10)
    plus_di = 100.0 * pd.Series(plus_dm / atr_safe).ewm(alpha=1 / 14, adjust=False).mean().values
    minus_di = 100.0 * pd.Series(minus_dm / atr_safe).ewm(alpha=1 / 14, adjust=False).mean().values
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    out["adx"] = pd.Series(dx).ewm(alpha=1 / 14, adjust=False).mean().values

    # Bollinger position 20
    sma = pd.Series(close).rolling(20).mean().values
    std = pd.Series(close).rolling(20).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_pos = (close - lower) / np.maximum(upper - lower, 1e-10)
    out["bb_position_20"] = np.clip(bb_pos, 0.0, 1.0)
    out["bb_width_20"] = (upper - lower) / np.maximum(sma, 1e-10)

    # Volume ratios
    vol_sma_10 = pd.Series(volume).rolling(10).mean().values
    vol_sma_20 = pd.Series(volume).rolling(20).mean().values
    out["volume_ratio_10"] = volume / np.maximum(vol_sma_10, 1e-10)
    out["volume_ratio_20"] = volume / np.maximum(vol_sma_20, 1e-10)

    # Volatility (rolling std of returns)
    rets = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-10)
    for w in (5, 10, 20):
        out[f"volatility_{w}"] = pd.Series(rets).rolling(w).std().values

    out["returns"] = rets
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# ─────────────────────────────────────────────────────────────────
# Model schema discovery
# ─────────────────────────────────────────────────────────────────

def load_model_feature_names(model_path: Path) -> Tuple[List[str], int]:
    """Return (feature_names, n_features) saved with the joint ridge model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(model_path, "rb") as fh:
            data = pickle.load(fh)
    feature_names = data.get("feature_names") or []
    n_features = data.get("n_features") or len(feature_names)
    if not feature_names:
        raise ValueError(f"{model_path} has no feature_names — cannot backfill")
    return list(feature_names), int(n_features)


# ─────────────────────────────────────────────────────────────────
# Per-pair OHLCV cache
# ─────────────────────────────────────────────────────────────────

def _newest_h1_csv(pair: str, market_dir: Path) -> Optional[Path]:
    csvs = sorted(
        market_dir.glob(f"oanda_{pair}_H1_live_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return csvs[0] if csvs else None


_OHLCV_CACHE: Dict[str, Optional[pd.DataFrame]] = {}


def get_features_for_pair(pair: str, market_dir: Path) -> Optional[pd.DataFrame]:
    """Load + cache (newest H1 CSV) features for a pair. Returns None if no CSV."""
    if pair in _OHLCV_CACHE:
        return _OHLCV_CACHE[pair]
    csv = _newest_h1_csv(pair, market_dir)
    if csv is None:
        logger.warning("No H1 CSV for %s in %s", pair, market_dir)
        _OHLCV_CACHE[pair] = None
        return None
    try:
        df = pd.read_csv(csv)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        if len(df) < 50:
            logger.warning("CSV for %s too short (%d rows)", pair, len(df))
            _OHLCV_CACHE[pair] = None
            return None
        feat = _add_indicators(df)
        _OHLCV_CACHE[pair] = feat
        logger.info("Loaded %s features: %d rows, %s -> %s",
                    pair, len(feat), feat.index[0], feat.index[-1])
        return feat
    except Exception as exc:
        logger.warning("Failed loading CSV for %s: %s", pair, exc)
        _OHLCV_CACHE[pair] = None
        return None


# ─────────────────────────────────────────────────────────────────
# Per-trade backfill
# ─────────────────────────────────────────────────────────────────

def _coerce_timestamp(ts: Any) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    try:
        out = pd.to_datetime(ts, utc=True)
        return out
    except Exception:
        return None


def derive_ridge_features(
    pair: str,
    timestamp: Any,
    feature_names: List[str],
    market_dir: Path = _DEFAULT_MARKET_DIR,
    max_bar_age_days: float = 180.0,
) -> Tuple[Optional[List[float]], Optional[str], Optional[float]]:
    """Re-derive a 24-feature vector for ``(pair, timestamp)``.

    Returns ``(features, skip_reason, bar_age_days)``. On success
    ``features`` is a list of length ``len(feature_names)``,
    ``skip_reason`` is None, and ``bar_age_days`` is the gap between
    the trade's timestamp and the OHLCV bar used. On skip, ``features``
    is None and ``skip_reason`` is one of: ``"no_csv"``,
    ``"timestamp_oob"``, ``"other:<msg>"``.

    The default ``max_bar_age_days=180`` is intentionally generous: this
    script's primary use-case is bridging the 2026-04-30 leak-fix gap,
    where live trades from April were logged against a model whose
    feature schema didn't yet exist and the only OHLCV CSVs available
    are from earlier in the year. The instrument one-hot (which is the
    dominant signal in this 24-feature schema for cross-pair generalisation)
    is recovered correctly regardless of bar age. Operators using fresh
    OHLCV should pass ``--max-bar-age-days=30`` for stricter validation.
    """
    df_feat = get_features_for_pair(pair, market_dir)
    if df_feat is None:
        return None, "no_csv", None

    ts = _coerce_timestamp(timestamp)
    if ts is None:
        return None, "other:bad_timestamp", None

    # Find the row at or just before `ts` (most-recent closed bar).
    valid_idx = df_feat.index[df_feat.index <= ts]
    if len(valid_idx) == 0:
        return None, "timestamp_oob", None
    row_ts = valid_idx[-1]
    age_days = (ts - row_ts).total_seconds() / 86400.0
    if age_days > max_bar_age_days:
        return None, "timestamp_oob", float(age_days)

    # Build single-row DataFrame
    last_row = df_feat.loc[[row_ts]]

    # Use the same builder gates.py uses for numeric features.
    try:
        aligned = align_features_to_model(last_row, feature_names, fill_value=0.0)
    except Exception as exc:
        return None, f"other:align_failed:{exc!s}", float(age_days)

    # `align_features_to_model` skips columns starting with `instrument_`
    # (filled with 0.0). Manually fill the matching one-hot. Mirrors
    # `gates.py:1263-1266`.
    vec = aligned[0].astype(np.float32).copy()
    for i, col in enumerate(feature_names):
        if col.startswith("instrument_"):
            expected = col.replace("instrument_", "", 1)
            vec[i] = 1.0 if expected == pair else 0.0

    # Sanitize: NaN/Inf → 0.0 (defensive — `_add_indicators` already does
    # this but stacking might re-introduce NaN at warm-up boundaries).
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec.tolist(), None, float(age_days)


def backfill_journal(
    journal_path: Path = _DEFAULT_JOURNAL,
    model_path: Path = _DEFAULT_MODEL,
    market_dir: Path = _DEFAULT_MARKET_DIR,
    dry_run: bool = False,
    max_bar_age_days: float = 180.0,
) -> BackfillSummary:
    """Read journal, backfill where needed, write atomically. Returns summary."""
    feature_names, n_expected = load_model_feature_names(model_path)
    logger.info("Model schema: n_features=%d, sample_names=%s...",
                n_expected, feature_names[:5])

    if not journal_path.exists():
        logger.error("Journal not found: %s", journal_path)
        return BackfillSummary()

    raw = journal_path.read_text()
    try:
        trades = json.loads(raw)
    except ValueError as exc:
        logger.error("Journal JSON parse failed: %s", exc)
        return BackfillSummary()
    if not isinstance(trades, list):
        logger.error("Journal is not a list (got %s)", type(trades).__name__)
        return BackfillSummary()

    summary = BackfillSummary(n_total=len(trades))

    for entry in trades:
        if not isinstance(entry, dict):
            summary.n_skipped_other += 1
            continue
        rf = entry.get("ridge_features")
        if isinstance(rf, list) and len(rf) == n_expected:
            summary.n_already_24 += 1
            continue

        pair = entry.get("pair")
        ts = entry.get("timestamp")
        if not pair or not ts:
            summary.n_skipped_other += 1
            summary.skip_details.append({
                "trade_id": entry.get("trade_id"),
                "reason": "missing_pair_or_timestamp",
            })
            continue

        new_vec, reason, bar_age = derive_ridge_features(
            pair, ts, feature_names, market_dir,
            max_bar_age_days=max_bar_age_days,
        )
        if new_vec is None:
            if reason == "no_csv":
                summary.n_skipped_no_csv += 1
            elif reason == "timestamp_oob":
                summary.n_skipped_timestamp_oob += 1
            else:
                summary.n_skipped_other += 1
            summary.skip_details.append({
                "trade_id": entry.get("trade_id"),
                "pair": pair,
                "timestamp": ts,
                "reason": reason,
                "bar_age_days": bar_age,
            })
            continue

        entry["ridge_features"] = new_vec
        # Tag with backfill metadata so downstream calibration can choose
        # to weight or drop trades whose feature row was stale.
        entry["ridge_features_backfill"] = {
            "bar_age_days": round(float(bar_age), 2) if bar_age is not None else None,
            "n_features": len(new_vec),
            "backfilled_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": "2026-04-30-leak-fix",
        }
        summary.n_backfilled += 1
        summary.pairs_touched.append(pair)

    if dry_run:
        logger.info("dry_run=True; NOT writing journal")
        return summary

    if summary.n_backfilled == 0:
        logger.info("No entries needed backfill — leaving journal untouched")
        return summary

    # Atomic write: tmp + rename, sha256 verified after rename.
    tmp = journal_path.with_suffix(journal_path.suffix + ".tmp")
    payload = json.dumps(trades, indent=2, sort_keys=False, default=str)
    tmp.write_text(payload)
    pre_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    os.replace(tmp, journal_path)
    post_sha = hashlib.sha256(journal_path.read_bytes()).hexdigest()
    if pre_sha != post_sha:
        # Should not happen; rename is atomic on POSIX.
        logger.error("sha256 mismatch after rename (pre=%s post=%s)", pre_sha, post_sha)
        raise RuntimeError("Atomic write integrity failure")
    logger.info(
        "Wrote backfilled journal: %s (sha256=%s, %d backfilled / %d total)",
        journal_path, post_sha[:12], summary.n_backfilled, summary.n_total,
    )
    return summary


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=_DEFAULT_JOURNAL)
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--market-dir", type=Path, default=_DEFAULT_MARKET_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="don't write — just report what would change")
    parser.add_argument("--max-bar-age-days", type=float, default=180.0,
                        help="max gap (in days) between trade timestamp and "
                             "the OHLCV bar used for feature derivation. "
                             "Default 180; tighten to 30 once OHLCV freshness "
                             "improves.")
    args = parser.parse_args()

    summary = backfill_journal(
        journal_path=args.journal,
        model_path=args.model,
        market_dir=args.market_dir,
        dry_run=args.dry_run,
        max_bar_age_days=args.max_bar_age_days,
    )
    logger.info("Summary: %s", json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
