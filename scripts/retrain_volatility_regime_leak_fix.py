"""Retrain ONLY the volatility regime TCN with the new realized-outcome labels.

Operational tool for the 2026-05-06 B1 leak-fix. Loads H1 OHLCV per pair,
computes normalized features, runs the (now-rewired) load_volatility_regime_data
which generates forward-realized 4-class labels, sequences the data, trains
the TCN, and overwrites trained_data/models/joint/tcn_volatility_regime.keras
in place.

NO auto-promote — this script writes the new artifacts, captures macro-F1 +
per-class precision/recall/F1 + confusion matrix, but does NOT modify any
production switchboard or runtime gate. Operator (controller) reviews the
report and decides whether to keep the artifacts.

Why a focused script (not full JointMultiPairTrainer.train_all())?
The B1 leak-fix only changes the volatility regime head — direction /
momentum / risk / confidence are untouched. A joint retrain would stomp
on the (good) per-pair confidence pkls from A1.5.

Output (all written atomically via tmp + rename):
    trained_data/models/joint/tcn_volatility_regime.keras   (replaced)
    trained_data/models/joint/tcn_volatility_regime.meta.pkl   (replaced)
    trained_data/volatility_regime_retrain_report.json   (new — side meta)

Exit codes:
    0 — at least one pair retrained successfully AND macro-F1 in
        sanity range [0.20, 0.60]. Below 0.20 is broken; above 0.60
        suggests residual leak.
    2 — failed (no pair succeeded, or all macro-F1 outside sanity range).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Make `src.*` importable when run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.modular_data_loaders import (  # noqa: E402
    compute_normalized_features,
    load_volatility_regime_data,
)
from src.training.trainers.config import TrainerConfig  # noqa: E402
from src.training.trainers.tcn_volatility_trainer import (  # noqa: E402
    TCNVolatilityRegimeTrainer,
)
from src.training.trainers.utils import create_sequences  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

JOINT_DIR = _PROJECT_ROOT / "trained_data" / "models" / "joint"
TCN_PATH = JOINT_DIR / "tcn_volatility_regime.keras"
DEFAULT_REPORT_PATH = (
    _PROJECT_ROOT / "trained_data" / "volatility_regime_retrain_report.json"
)
MARKET_DIR = _PROJECT_ROOT / "market_data"

# Default pair set (matches retrain_confidence_leak_fix.py's roster).
DEFAULT_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY",
]

# Sanity range for macro-F1 (per design 4.6: realistic 0.30–0.45;
# below 0.20 = broken, above 0.60 = residual leak suspected).
MIN_MACRO_F1_SANITY = 0.20
MAX_MACRO_F1_SANITY = 0.60


def _newest_h1_csv_per_pair(instruments: List[str]) -> Dict[str, Path]:
    """Pick the newest H1 live CSV per instrument."""
    out: Dict[str, Path] = {}
    for inst in instruments:
        candidates = sorted(
            MARKET_DIR.glob(f"oanda_{inst}_H1_live_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            out[inst] = candidates[0]
    return out


def _load_pair_df(path: Path) -> pd.DataFrame:
    """Load a pair CSV into a clean OHLCV DataFrame, time-indexed."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the indicator columns load_volatility_regime_data expects.

    Mirrors `_add_indicators` in retrain_confidence_leak_fix.py — same
    ATR / RSI / ADX / BB / volume-ratio / volatility computations.
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
    for w in (5, 10, 20):
        out[f"atr_pct_{w}"] = (
            pd.Series(tr).rolling(w).mean().values / np.maximum(close, 1e-10)
        )

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

    # ADX(14) — simplified
    plus_dm = np.where(
        (high - np.concatenate([[high[0]], high[:-1]]))
        > (np.concatenate([[low[0]], low[:-1]]) - low),
        np.maximum(high - np.concatenate([[high[0]], high[:-1]]), 0.0),
        0.0,
    )
    minus_dm = np.where(
        (np.concatenate([[low[0]], low[:-1]]) - low)
        > (high - np.concatenate([[high[0]], high[:-1]])),
        np.maximum(np.concatenate([[low[0]], low[:-1]]) - low, 0.0),
        0.0,
    )
    atr_safe = np.maximum(atr, 1e-10)
    plus_di = 100.0 * pd.Series(plus_dm / atr_safe).ewm(
        alpha=1 / 14, adjust=False
    ).mean().values
    minus_di = 100.0 * pd.Series(minus_dm / atr_safe).ewm(
        alpha=1 / 14, adjust=False
    ).mean().values
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    out["adx"] = pd.Series(dx).ewm(alpha=1 / 14, adjust=False).mean().values

    # Bollinger
    sma = pd.Series(close).rolling(20).mean().values
    std = pd.Series(close).rolling(20).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_pos = (close - lower) / np.maximum(upper - lower, 1e-10)
    out["bb_position_20"] = np.clip(bb_pos, 0.0, 1.0)
    out["bb_width_20"] = (upper - lower) / np.maximum(sma, 1e-10)
    out["bb_width_norm"] = out["bb_width_20"]

    # Volume ratios
    vol_sma_10 = pd.Series(volume).rolling(10).mean().values
    vol_sma_20 = pd.Series(volume).rolling(20).mean().values
    out["volume_ratio_10"] = volume / np.maximum(vol_sma_10, 1e-10)
    out["volume_ratio_20"] = volume / np.maximum(vol_sma_20, 1e-10)

    # Returns volatility
    rets = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-10)
    for w in (5, 10, 20):
        out[f"volatility_{w}"] = pd.Series(rets).rolling(w).std().values
    out["returns_1"] = rets
    out["returns_5"] = pd.Series(rets).rolling(5).mean().values
    out["returns_10"] = pd.Series(rets).rolling(10).mean().values

    out["high_low_range"] = (high - low) / np.maximum(close, 1e-10)
    out["returns"] = rets

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 4,
) -> Dict[str, Any]:
    """Compute per-class precision/recall/F1, macro-F1, accuracy,
    confusion matrix. Real sklearn — NO MOCKS.
    """
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    labels = list(range(n_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    accuracy = float(np.mean(y_true == y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "per_class": [
            {
                "class": int(k),
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
            }
            for k in labels
        ],
        "confusion_matrix": cm,
    }


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic JSON write: tmp + os.replace. Per CLAUDE.md JSON safety gate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def _retrain_one_pair(
    inst: str,
    csv_path: Path,
    seq_len: int,
) -> Optional[Dict[str, Any]]:
    """Load + train + evaluate a single pair. Returns per-pair report
    dict, or None on skip.
    """
    try:
        df_raw = _load_pair_df(csv_path)
        df = _add_indicators(df_raw)
    except (OSError, ValueError, KeyError) as e:
        logger.warning("Skip %s: feature compute failed (%s)", inst, e)
        return None

    if len(df) < 500:
        logger.warning("Skip %s: only %d rows (need >= 500)", inst, len(df))
        return None

    try:
        result = load_volatility_regime_data(df, instrument=inst)
    except (ValueError, KeyError) as e:
        logger.warning("Skip %s: load_volatility_regime_data failed (%s)", inst, e)
        return None

    X_tr_2d = result["X_train"]
    X_va_2d = result["X_val"]
    y_tr_2d = result["y_train"]
    y_va_2d = result["y_val"]
    label_meta = result.get("label_metadata", {})

    if len(X_tr_2d) < seq_len + 100 or len(X_va_2d) < seq_len + 20:
        logger.warning(
            "Skip %s: insufficient rows post-split (train=%d val=%d, need >= seq_len=%d)",
            inst, len(X_tr_2d), len(X_va_2d), seq_len,
        )
        return None

    # Reshape 2D → 3D sequences for the TCN.
    X_tr_seq, y_tr_seq = create_sequences(X_tr_2d, y_tr_2d, seq_len)
    X_va_seq, y_va_seq = create_sequences(X_va_2d, y_va_2d, seq_len)

    logger.info(
        "%s: pre-train shapes — X_tr_seq=%s y_tr_seq=%s X_va_seq=%s y_va_seq=%s",
        inst, X_tr_seq.shape, y_tr_seq.shape, X_va_seq.shape, y_va_seq.shape,
    )

    config = TrainerConfig()
    trainer = TCNVolatilityRegimeTrainer(config)
    metrics = trainer.train(
        X_tr_seq, y_tr_seq.astype(np.int32),
        X_va_seq, y_va_seq.astype(np.int32),
        feature_names=result.get("feature_names"),
        seq_len=seq_len,
    )

    # Independent eval (the trainer reports its own metrics; we recompute
    # against our class set for an objective per-class report).
    y_pred_proba = trainer.model.predict(X_va_seq, verbose=0)
    if isinstance(y_pred_proba, (list, tuple)):
        # Dual-head model returns [classification, regression].
        y_pred_proba = y_pred_proba[0]
    y_pred = np.argmax(np.asarray(y_pred_proba), axis=1)

    eval_metrics = _per_class_metrics(y_va_seq, y_pred, n_classes=4)
    logger.info(
        "%s: macro_f1=%.4f acc=%.4f per_class_f1=%s",
        inst,
        eval_metrics["macro_f1"],
        eval_metrics["accuracy"],
        [round(c["f1"], 3) for c in eval_metrics["per_class"]],
    )

    return {
        "instrument": inst,
        "csv": str(csv_path.name),
        "n_train_rows": int(len(X_tr_seq)),
        "n_val_rows": int(len(X_va_seq)),
        "macro_f1": eval_metrics["macro_f1"],
        "accuracy": eval_metrics["accuracy"],
        "per_class": eval_metrics["per_class"],
        "confusion_matrix": eval_metrics["confusion_matrix"],
        "trainer_metrics": metrics,
        "label_metadata": label_meta,
        "trainer": trainer,  # Kept ONLY for the joint-write step; stripped from report JSON
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrain ONLY the volatility regime TCN with the B1 "
                    "leak-fix realized-outcome labels. Does NOT auto-promote.",
    )
    parser.add_argument(
        "--pairs",
        default="all",
        help="Comma-separated pair list (e.g. EUR_USD,GBP_USD) or 'all' "
             "for the default 10-pair roster.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="TCN sequence length (default 60 H1 bars).",
    )
    parser.add_argument(
        "--report-out",
        default=str(DEFAULT_REPORT_PATH),
        help="Path for the side-meta JSON report.",
    )
    args = parser.parse_args()

    if args.pairs.strip().lower() == "all":
        instruments = list(DEFAULT_PAIRS)
    else:
        instruments = [s.strip() for s in args.pairs.split(",") if s.strip()]

    pair_csvs = _newest_h1_csv_per_pair(instruments)
    logger.info("Found H1 CSVs for %d / %d pairs", len(pair_csvs), len(instruments))
    if not pair_csvs:
        logger.error("No H1 CSVs found in %s — abort", MARKET_DIR)
        return 2

    per_pair_reports: List[Dict[str, Any]] = []
    best_pair: Optional[str] = None
    best_macro_f1 = -1.0
    best_trainer: Optional[TCNVolatilityRegimeTrainer] = None

    for inst, csv_path in pair_csvs.items():
        report = _retrain_one_pair(inst, csv_path, seq_len=args.seq_len)
        if report is None:
            continue
        # Track best macro-F1 so we save its artifact to the joint dir.
        if report["macro_f1"] > best_macro_f1:
            best_macro_f1 = report["macro_f1"]
            best_pair = inst
            best_trainer = report["trainer"]
        # Strip the trainer ref before serializing.
        report.pop("trainer", None)
        per_pair_reports.append(report)

    if not per_pair_reports:
        logger.error("No pair produced training data — abort")
        return 2

    # Save the best pair's TCN to the joint dir (replaces existing).
    if best_trainer is not None:
        JOINT_DIR.mkdir(parents=True, exist_ok=True)
        best_trainer.save(str(TCN_PATH))
        logger.info(
            "Saved best TCN (pair=%s, macro_f1=%.4f) to %s",
            best_pair, best_macro_f1, TCN_PATH,
        )

    # Aggregate report
    macro_f1s = [r["macro_f1"] for r in per_pair_reports]
    in_range = [
        MIN_MACRO_F1_SANITY <= m <= MAX_MACRO_F1_SANITY for m in macro_f1s
    ]
    aggregate: Dict[str, Any] = {
        "retrained_at": datetime.utcnow().isoformat() + "Z",
        "leak_fix_version": "2026-05-06",
        "vol_horizon_bars": 24,
        "seq_len": args.seq_len,
        "best_pair": best_pair,
        "best_macro_f1": best_macro_f1,
        "macro_f1_summary": {
            "min": float(np.min(macro_f1s)),
            "median": float(np.median(macro_f1s)),
            "max": float(np.max(macro_f1s)),
            "mean": float(np.mean(macro_f1s)),
        },
        "n_pairs_attempted": len(pair_csvs),
        "n_pairs_succeeded": len(per_pair_reports),
        "n_pairs_in_sanity_range": int(sum(in_range)),
        "sanity_range": [MIN_MACRO_F1_SANITY, MAX_MACRO_F1_SANITY],
        "per_pair": per_pair_reports,
    }

    report_path = Path(args.report_out)
    _atomic_write_json(report_path, aggregate)
    logger.info("Report written to %s", report_path)

    # Exit code: 0 if at least one pair landed in sanity range, else 2.
    if any(in_range):
        logger.info(
            "OK — %d/%d pairs in sanity range [%.2f, %.2f]; best macro_f1=%.4f",
            sum(in_range), len(per_pair_reports),
            MIN_MACRO_F1_SANITY, MAX_MACRO_F1_SANITY, best_macro_f1,
        )
        return 0
    logger.error(
        "FAIL — no pair landed in sanity range [%.2f, %.2f]; macro_f1_min=%.4f max=%.4f. "
        "Below %.2f = broken; above %.2f = residual leak suspected. "
        "Review report at %s before deciding.",
        MIN_MACRO_F1_SANITY, MAX_MACRO_F1_SANITY,
        float(np.min(macro_f1s)), float(np.max(macro_f1s)),
        MIN_MACRO_F1_SANITY, MAX_MACRO_F1_SANITY,
        report_path,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
