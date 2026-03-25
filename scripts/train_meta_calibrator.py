#!/usr/bin/env python3
"""
Meta-Labeler Retraining + Confidence Calibrator — M1 Mac optimized.

Tier 4: Meta-Labeler — learns WHEN to trust primary model signals.
Tier 5: Confidence Calibrator (Platt scaling) — calibrates raw confidence to P(win).

Usage:
    python scripts/train_meta_calibrator.py
    python scripts/train_meta_calibrator.py --skip-meta
    python scripts/train_meta_calibrator.py --skip-calibrator
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

env_path = Path(__file__).resolve().parent.parent / ".env.local"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("meta_calibrator")

CONFIG_PATH = str(PROJECT_ROOT / "config" / "config_m1_optimized.yaml")
MODELS_DIR = PROJECT_ROOT / "trained_data" / "models"
JOINT_DIR = MODELS_DIR / "joint"
CACHE_DIR = PROJECT_ROOT / "trained_data" / "cache" / "training_data"
JOURNAL_PATH = PROJECT_ROOT / "trained_data" / "trade_journal.json"
JOURNAL_RL_PATH = PROJECT_ROOT / "trained_data" / "trade_journal_rl.json"

MASTER_PAIRS = ["GBP_JPY", "EUR_GBP", "AUD_USD", "USD_CAD", "AUD_NZD"]


def generate_simulated_trades(pairs: list, n_per_pair: int = 2000) -> list:
    """
    Generate simulated trade samples from trained models for meta-labeler training.

    When no real trade journal exists, we simulate trades by running ensemble
    predictions against held-out validation data and computing outcomes.
    """
    from src.utils import load_config
    from src.data.feature_engineering import FeatureEngineering
    from src.core.modular_data_loaders import load_direction_data

    cfg = load_config(CONFIG_PATH)
    fe = FeatureEngineering(cfg)
    samples = []

    for pair in pairs:
        cache_file = CACHE_DIR / f"{pair}_H1_25000.csv"
        if not cache_file.exists():
            logger.warning(f"No cached data for {pair}, skipping")
            continue

        df = pd.read_csv(cache_file, parse_dates=["time"])
        df_feat = fe.create_features(df.copy(), include_all=True)

        # Load direction data for validation split
        dir_data = load_direction_data(df_feat)
        if not dir_data:
            continue

        X_val = dir_data["X_val"]
        y_val = dir_data["y_val"]

        # Load trained models to get predictions
        model_dir = MODELS_DIR / pair

        # Get transformer predictions
        trans_probs = np.full(len(y_val), 0.5)
        trans_path = model_dir / "transformer_direction.keras"
        if trans_path.exists():
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(str(trans_path), compile=False)
                trans_probs = model.predict(X_val, verbose=0, batch_size=256).flatten()
                tf.keras.backend.clear_session()
            except Exception as e:
                logger.warning(f"Could not load transformer for {pair}: {e}")

        # Get ridge confidence
        ridge_confs = np.full(len(y_val), 50.0)
        ridge_path = model_dir / "ridge_confidence.pkl"
        if ridge_path.exists():
            try:
                import pickle
                with open(ridge_path, "rb") as f:
                    ridge_model = pickle.load(f)
                X_val_flat = X_val.reshape(X_val.shape[0], -1) if X_val.ndim == 3 else X_val
                if hasattr(ridge_model, "predict"):
                    ridge_confs = ridge_model.predict(X_val_flat) * 100
            except Exception:
                pass

        # Create trade samples
        for i in range(min(len(y_val), n_per_pair)):
            direction = "long" if trans_probs[i] > 0.5 else "short"
            predicted_dir = 1 if trans_probs[i] > 0.5 else 0
            actual_dir = int(y_val[i])
            correct = predicted_dir == actual_dir

            # Simulate PnL
            pnl = np.random.uniform(0.5, 3.0) if correct else np.random.uniform(-3.0, -0.5)

            samples.append({
                "tcn_probability": float(trans_probs[i]),
                "ridge_confidence": float(ridge_confs[i]),
                "xgb_momentum": float(np.random.uniform(0.3, 0.7)),  # placeholder
                "rf_drawdown_pips": float(np.random.uniform(0, 50)),  # placeholder
                "direction": direction,
                "instrument": pair,
                "actual_outcome": "win" if correct else "loss",
                "pnl": pnl,
                "pnl_pips": pnl * 10,
            })

        del df, df_feat, dir_data
        gc.collect()

    logger.info(f"Generated {len(samples)} simulated trade samples")
    return samples


def train_meta_labeler() -> dict:
    """Train the meta-labeler model."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING: Meta-Labeler (XGBoost)")
    logger.info("="*60)

    from src.training.meta_labeler_retrainer import MetaLabelerRetrainer, MetaLabelerTrainingConfig, TradeSample

    config = MetaLabelerTrainingConfig(
        use_xgboost=True,
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        reg_alpha=0.5,
        reg_lambda=3.0,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        early_stopping_rounds=25,
        max_overfitting_gap=0.08,
        min_confidence_threshold=0.50,
    )

    retrainer = MetaLabelerRetrainer(
        config=config,
        model_dir=JOINT_DIR,
        output_dir=JOINT_DIR,
    )

    # Try loading real trade journal first
    trade_samples = []
    for journal_path in [JOURNAL_PATH, JOURNAL_RL_PATH]:
        if journal_path.exists():
            loaded = retrainer.load_trade_journal(journal_path)
            if loaded:
                trade_samples.extend(loaded)
                logger.info(f"Loaded {len(loaded)} trades from {journal_path}")

    # If not enough real trades, generate simulated ones
    if len(trade_samples) < 100:
        logger.info("Not enough real trades — generating simulated trade data...")
        sim_data = generate_simulated_trades(MASTER_PAIRS)

        trade_samples = [
            TradeSample(
                tcn_probability=s["tcn_probability"],
                ridge_confidence=s["ridge_confidence"],
                xgb_momentum=s["xgb_momentum"],
                rf_drawdown_pips=s["rf_drawdown_pips"],
                direction=s["direction"],
                instrument=s.get("instrument", ""),
                actual_outcome=s.get("actual_outcome", ""),
                pnl=s.get("pnl", 0.0),
                pnl_pips=s.get("pnl_pips", 0.0),
            )
            for s in sim_data
        ]

    if len(trade_samples) < 50:
        logger.warning("Still not enough data for meta-labeler training")
        return {"status": "skipped", "reason": "Insufficient trade data"}

    # Build feature matrix
    features = np.array([
        [s.tcn_probability, s.ridge_confidence, s.xgb_momentum, s.rf_drawdown_pips]
        for s in trade_samples
    ])
    labels = np.array([1 if s.actual_outcome == "win" else 0 for s in trade_samples])

    # Train/val split
    split_idx = int(len(features) * 0.8)
    X_train, X_val = features[:split_idx], features[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]

    try:
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate,
            "n_estimators": config.n_estimators,
            "reg_alpha": config.reg_alpha,
            "reg_lambda": config.reg_lambda,
            "subsample": config.subsample,
            "colsample_bytree": config.colsample_bytree,
            "min_child_weight": config.min_child_weight,
            "verbosity": 0,
        }

        model = xgb.train(
            params, dtrain,
            num_boost_round=config.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=25,
        )

        # Evaluate
        train_preds = model.predict(dtrain)
        val_preds = model.predict(dval)
        train_acc = np.mean((train_preds > 0.5) == y_train)
        val_acc = np.mean((val_preds > 0.5) == y_val)
        gap = abs(train_acc - val_acc)

        # Save
        JOINT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = JOINT_DIR / "meta_labeler.pkl"
        import pickle
        with open(save_path, "wb") as f:
            pickle.dump({"model": model, "feature_names": ["tcn_prob", "ridge_conf", "xgb_mom", "rf_dd"]}, f)

        logger.info(f"✓ Meta-Labeler: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, gap={gap:.4f}")
        return {
            "status": "success",
            "train_acc": train_acc, "val_acc": val_acc, "gap": gap,
            "n_samples": len(features),
        }

    except ImportError:
        logger.warning("XGBoost not installed — using sklearn fallback")
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
        )
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        JOINT_DIR.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(JOINT_DIR / "meta_labeler.pkl", "wb") as f:
            pickle.dump({"model": model, "feature_names": ["tcn_prob", "ridge_conf", "xgb_mom", "rf_dd"]}, f)

        return {"status": "success", "train_acc": train_acc, "val_acc": val_acc}

    except Exception as e:
        logger.error(f"✗ Meta-Labeler failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}


def train_confidence_calibrator() -> dict:
    """Train Platt scaling confidence calibrator."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING: Confidence Calibrator (Platt Scaling)")
    logger.info("="*60)

    try:
        from src.scanner.automation.confidence_calibrator import ConfidenceCalibrator

        calibrator = ConfidenceCalibrator()

        # Try fitting from real journal
        for journal_path in [JOURNAL_PATH, JOURNAL_RL_PATH]:
            if journal_path.exists():
                try:
                    calibrator.fit_from_journal(str(journal_path))
                    if calibrator.is_fitted:
                        calibrator.save()
                        logger.info(f"✓ Calibrator fitted from {journal_path}")
                        return {"status": "success", "source": str(journal_path)}
                except Exception as e:
                    logger.warning(f"Could not fit from {journal_path}: {e}")

        # If no journal, generate simulated data
        logger.info("No trade journal — fitting on simulated confidence data...")
        sim_data = generate_simulated_trades(MASTER_PAIRS, n_per_pair=500)
        if sim_data:
            raw_scores = np.array([s["tcn_probability"] for s in sim_data])
            outcomes = np.array([1 if s["actual_outcome"] == "win" else 0 for s in sim_data])

            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            lr.fit(raw_scores.reshape(-1, 1), outcomes)

            A = float(lr.coef_[0][0])
            B = float(lr.intercept_[0])

            # Save calibration params
            cal_path = MODELS_DIR / "confidence_calibration.json"
            cal_data = {
                "platt_A": A, "platt_B": B,
                "n_samples": len(raw_scores),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(cal_path, "w") as f:
                json.dump(cal_data, f, indent=2)

            logger.info(f"✓ Calibrator: A={A:.4f}, B={B:.4f} (Platt sigmoid)")
            return {"status": "success", "platt_A": A, "platt_B": B, "n_samples": len(raw_scores)}

        return {"status": "skipped", "reason": "No data available"}

    except Exception as e:
        logger.error(f"✗ Calibrator failed: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Train meta-labeler and calibrator")
    parser.add_argument("--skip-meta", action="store_true", help="Skip meta-labeler")
    parser.add_argument("--skip-calibrator", action="store_true", help="Skip calibrator")
    args = parser.parse_args()

    t0 = time.time()
    results = {}

    if not args.skip_meta:
        results["meta_labeler"] = train_meta_labeler()
        gc.collect()

    if not args.skip_calibrator:
        results["confidence_calibrator"] = train_confidence_calibrator()
        gc.collect()

    total_time = round(time.time() - t0, 1)
    report_path = MODELS_DIR / "meta_calibrator_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "results": results,
            "total_duration_s": total_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, sort_keys=True, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"META + CALIBRATOR COMPLETE — {total_time}s")
    for comp, r in results.items():
        logger.info(f"  {comp}: {r.get('status', 'unknown')}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
