"""Retrain ONLY the confidence head with the new realized-outcome labels.

This is the operational tool for the 2026-04-30 leak-fix PR. It loads
H1 OHLCV history per pair, computes normalized features, runs the new
`load_ridge_data` against pooled trade journal (live + archives), trains
a single joint LightGBM regressor on the rescaled binary labels, and
overwrites `trained_data/models/joint/ridge_confidence.pkl` in place.

Why a focused script (not full `JointMultiPairTrainer.train_all()`)?
The leak-fix PR only changes the confidence head — direction/momentum/
risk models are untouched. A full re-run would need the full ML stack
(transformer, TCN, LightGBM heads) and is out of scope.

Output:
    trained_data/models/joint/ridge_confidence.pkl   (replaced in-place)
    trained_data/models/joint/joint_training_meta.json   (confidence metrics
        and label_metadata replaced; direction/momentum/risk metrics preserved)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Make `src.*` importable when this script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.modular_data_loaders import (  # noqa: E402
    append_instrument_features,
    compute_normalized_features,
    load_ridge_data,
)
from src.training.trainers.config import TrainerConfig  # noqa: E402
from src.training.trainers.ridge_trainer import RidgeTrainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

JOINT_DIR = _PROJECT_ROOT / "trained_data" / "models" / "joint"
RIDGE_PATH = JOINT_DIR / "ridge_confidence.pkl"
META_PATH = JOINT_DIR / "joint_training_meta.json"
MARKET_DIR = _PROJECT_ROOT / "market_data"


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
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    # Drop OOS columns / dupes
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR-family + RSI/ADX/BB columns the loader expects.
    `compute_normalized_features` adds many but not all (e.g. atr, rsi, adx,
    bb_position_20). We compute these directly here.
    """
    out = compute_normalized_features(df).copy()
    high = out["high"].values
    low = out["low"].values
    close = out["close"].values
    volume = out["volume"].values

    # ATR(14) — true range exponential
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - np.concatenate([[close[0]], close[:-1]])),
        np.abs(low - np.concatenate([[close[0]], close[:-1]])),
    ])
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    out["atr"] = atr
    out["atr_pct_14"] = atr / np.maximum(close, 1e-10)

    # RSI(14) — Wilder
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
    plus_dm = np.where((high - np.concatenate([[high[0]], high[:-1]])) >
                       (np.concatenate([[low[0]], low[:-1]]) - low),
                       np.maximum(high - np.concatenate([[high[0]], high[:-1]]), 0.0), 0.0)
    minus_dm = np.where((np.concatenate([[low[0]], low[:-1]]) - low) >
                        (high - np.concatenate([[high[0]], high[:-1]])),
                        np.maximum(np.concatenate([[low[0]], low[:-1]]) - low, 0.0), 0.0)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instruments",
        default="EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD,USD_CAD,NZD_USD,EUR_GBP,EUR_JPY,GBP_JPY",
    )
    parser.add_argument("--lookahead-bars", type=int, default=24)
    parser.add_argument("--sl-atr-mult", type=float, default=1.0)
    parser.add_argument("--tp-atr-mult", type=float, default=2.5)
    parser.add_argument("--label-mode", default="blend")
    args = parser.parse_args()

    instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]
    pair_csvs = _newest_h1_csv_per_pair(instruments)
    logger.info("Found H1 CSVs for %d pairs", len(pair_csvs))

    if not pair_csvs:
        logger.error("No H1 CSVs found in %s — abort", MARKET_DIR)
        return 1

    # Load each pair's df, compute features, run loader, append instrument one-hots
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    feature_names = None
    label_meta_agg = {
        "n_real": 0, "n_pseudo": 0, "n_unavailable": 0,
        "n_dropped_nan": 0, "per_instrument": {},
    }

    for inst, csv_path in pair_csvs.items():
        try:
            df_raw = _load_pair_df(csv_path)
            df = _add_indicators(df_raw)
        except Exception as e:
            logger.warning("Skip %s: feature compute failed (%s)", inst, e)
            continue

        if len(df) < 200:
            logger.warning("Skip %s: only %d rows", inst, len(df))
            continue

        try:
            result = load_ridge_data(
                df,
                instrument=inst,
                label_mode=args.label_mode,
                lookahead_bars=args.lookahead_bars,
                sl_atr_mult=args.sl_atr_mult,
                tp_atr_mult=args.tp_atr_mult,
            )
        except Exception as e:
            logger.warning("Skip %s: load_ridge_data failed (%s)", inst, e)
            continue

        x_tr, fn = append_instrument_features(
            result["X_train"], inst, instruments,
            original_feature_names=result.get("feature_names"),
        )
        x_va, _ = append_instrument_features(
            result["X_val"], inst, instruments,
            original_feature_names=result.get("feature_names"),
        )
        X_train_list.append(x_tr)
        X_val_list.append(x_va)
        y_train_list.append(result["y_train"])
        y_val_list.append(result["y_val"])
        if feature_names is None:
            feature_names = fn

        lm = result.get("label_metadata") or {}
        label_meta_agg["n_real"] += int(lm.get("n_real", 0) or 0)
        label_meta_agg["n_pseudo"] += int(lm.get("n_pseudo", 0) or 0)
        label_meta_agg["n_unavailable"] += int(lm.get("n_unavailable", 0) or 0)
        label_meta_agg["n_dropped_nan"] += int(lm.get("n_dropped_nan", 0) or 0)
        label_meta_agg["per_instrument"][inst] = {
            "n_real": lm.get("n_real"),
            "n_pseudo": lm.get("n_pseudo"),
            "class_balance_real": lm.get("class_balance_real"),
            "class_balance_pseudo": lm.get("class_balance_pseudo"),
            "n_train": len(result["y_train"]),
            "n_val": len(result["y_val"]),
        }
        label_meta_agg["label_mode"] = lm.get("label_mode")
        label_meta_agg["score_range"] = lm.get("score_range")
        label_meta_agg["score_formula"] = lm.get("score_formula")
        label_meta_agg["leak_fix_version"] = lm.get("leak_fix_version")
        logger.info(
            "%s: train=%d val=%d n_real=%s n_pseudo=%s",
            inst, len(result["y_train"]), len(result["y_val"]),
            lm.get("n_real"), lm.get("n_pseudo"),
        )

    if not X_train_list:
        logger.error("No pairs produced training data — abort")
        return 2

    X_train = np.vstack(X_train_list)
    X_val = np.vstack(X_val_list)
    y_train = np.concatenate(y_train_list)
    y_val = np.concatenate(y_val_list)
    logger.info("Joint shape: train=%s val=%s", X_train.shape, X_val.shape)
    logger.info(
        "Aggregated labels: n_real=%d n_pseudo=%d total_train_y=%d (wins=%d)",
        label_meta_agg["n_real"], label_meta_agg["n_pseudo"],
        len(y_train), int(np.sum(y_train > 50.0)),
    )

    # Pull W&B control-plane config for the confidence head and apply it.
    cp_cfg: Dict = {}
    try:
        from src.training.wandb_control_plane import (
            apply_config_to_trainer,
            pull_config,
            seed_default,
        )
        # Idempotent first-time setup — uploads defaults if no artifact yet.
        seed_default("confidence")
        cp_cfg = pull_config("confidence")
    except Exception as exc:  # pragma: no cover — observability only
        logger.warning("W&B control-plane pull failed: %s — using script defaults", exc)
        cp_cfg = {}

    # Train
    config = TrainerConfig()
    trainer = RidgeTrainer(config)
    if cp_cfg:
        try:
            apply_config_to_trainer(trainer, cp_cfg)
            hps = cp_cfg.get("hyperparameters") or {}
            logger.info(
                "Applied W&B control-plane HPs: n_estimators=%s lr=%s max_depth=%s",
                hps.get("n_estimators"), hps.get("learning_rate"), hps.get("max_depth"),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("apply_config_to_trainer failed: %s", exc)
    metrics = trainer.train(
        X_train, y_train, X_val, y_val,
        feature_names=feature_names,
        label_metadata=label_meta_agg,
    )
    logger.info("Training metrics: %s", metrics)

    # Save
    JOINT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save(str(RIDGE_PATH))
    logger.info("Saved %s", RIDGE_PATH)

    # W&B logging — confidence-specific run (legacy detailed payload).
    try:
        from src.training.wandb_confidence import log_confidence_training_run
        wb_result = log_confidence_training_run(
            run_name="confidence_joint",
            metrics=metrics,
            label_metadata=label_meta_agg,
            model_path=str(RIDGE_PATH),
            config={
                "instruments": instruments,
                "label_mode": args.label_mode,
                "lookahead_bars": args.lookahead_bars,
                "sl_atr_mult": args.sl_atr_mult,
                "tp_atr_mult": args.tp_atr_mult,
            },
            tags=["joint", "leak_fix_2026_04_30", "source=manual"],
        )
        if wb_result:
            logger.info("W&B run: %s", wb_result.get("run_url") or wb_result.get("name"))
    except Exception as exc:
        logger.warning("W&B joint logging failed (non-fatal): %s", exc)

    # Control-plane run log — head=confidence, source=manual.
    try:
        from src.training.wandb_control_plane import log_run
        cp_result = log_run(
            head="confidence",
            config=cp_cfg or {},
            metrics={
                "val_r2": metrics.get("r2_score"),
                "val_mae": metrics.get("confidence_mae"),
                "n_real": (label_meta_agg or {}).get("n_real"),
                "n_pseudo": (label_meta_agg or {}).get("n_pseudo"),
            },
            artifacts=[Path(RIDGE_PATH)],
            run_name=f"manual_confidence_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            tags=["leak_fix_2026_04_30"],
            extra_config={"source": "manual", "script": "retrain_confidence_leak_fix"},
        )
        if cp_result:
            logger.info("Control-plane run: %s", cp_result.get("run_url") or cp_result.get("name"))
    except Exception as exc:
        logger.warning("Control-plane log_run failed (non-fatal): %s", exc)

    # Update joint_training_meta.json (preserve other heads)
    if META_PATH.exists():
        try:
            existing = json.loads(META_PATH.read_text())
        except Exception:
            existing = {}
    else:
        existing = {}

    existing.setdefault("metrics", {})
    existing["metrics"]["confidence"] = metrics
    existing["confidence_label_metadata"] = label_meta_agg
    existing["confidence_retrained_at"] = datetime.utcnow().isoformat() + "Z"
    existing.setdefault("instruments", instruments)

    tmp = META_PATH.with_suffix(META_PATH.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, META_PATH)
    logger.info("Updated %s", META_PATH)

    # Quick distribution check
    y_pred = trainer.model.predict(
        pd.DataFrame(trainer.scaler.transform(X_val), columns=feature_names)
    )
    logger.info(
        "Val score distribution: min=%.2f p25=%.2f p50=%.2f p75=%.2f max=%.2f",
        float(np.min(y_pred)), float(np.percentile(y_pred, 25)),
        float(np.percentile(y_pred, 50)), float(np.percentile(y_pred, 75)),
        float(np.max(y_pred)),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
