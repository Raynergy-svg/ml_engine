#!/usr/bin/env python3
"""
Train a single model for a single instrument — M1 Mac tf-metal optimized.

Restores full wandb-optimized hyperparameters (seq_len 120/90, batch_size 128)
since M1 has ample unified memory vs the constrained VM.

Usage:
    python scripts/train_single_model_m1.py --instrument GBP_JPY --model transformer --candles 25000
    python scripts/train_single_model_m1.py --instrument GBP_JPY --model all --candles 25000
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── tf-metal configuration ───────────────────────────────────────────────────
# Must be set BEFORE importing TensorFlow
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# M1 Metal GPU acceleration — let TF find the Metal plugin automatically
# If you hit Metal issues, uncomment: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load .env.local
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_m1")

CONFIG_PATH = str(PROJECT_ROOT / "config" / "config_m1_optimized.yaml")
MODELS_DIR = PROJECT_ROOT / "trained_data" / "models"
CACHE_DIR = PROJECT_ROOT / "trained_data" / "cache" / "training_data"
MAX_GAP = 0.06         # 6% — PASS/FAIL threshold for RESULT logging (cosmetic).
HARD_MAX_GAP = 0.10    # 10% — operator ship rule (2026-05-13 directive):
                       # "no models is to be shipped higher then 10% gap".
                       # Models that violate are moved to a quarantine dir so
                       # per-pair gate routing cannot pick them up.


def _read_trainer_metrics(trainer) -> dict:
    """Read the trainer's final metrics dict.

    Transformer/HistGB trainers store the dict on ``self.metrics`` after
    ``train()`` returns. The legacy ``get_metrics()`` accessor doesn't
    exist on TransformerDirectionTrainer — previously every caller read
    ``getattr(trainer, "get_metrics", lambda: {})()`` which silently
    returned ``{}`` and zeroed train_acc/val_acc/gap → the 6% PASS gate
    became a permanent no-op.
    """
    m = getattr(trainer, "metrics", None)
    if isinstance(m, dict) and m:
        return m
    if hasattr(trainer, "get_metrics"):
        try:
            got = trainer.get_metrics() or {}
            return got if isinstance(got, dict) else {}
        except Exception:
            pass
    return {}


def _snapshot_existing_artifacts(model_path) -> "Path | None":
    """Copy the current live artifacts sharing a model's stem into a rollback
    dir BEFORE a retrain overwrites them.

    Returns the rollback dir so a later quarantine (gap > HARD_MAX_GAP) can
    restore the prior known-good model to the live dir, or None if no live
    artifacts exist yet. Copies (never moves) so the in-progress retrain still
    sees its own working files; the rollback dir lives in a `_rollback/`
    subdir whose name does NOT share the model stem, so the quarantine glob
    below never sweeps it up.
    """
    path = Path(model_path)
    siblings = [p for p in path.parent.glob(f"{path.stem}*") if p.is_file()]
    if not siblings:
        return None
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rollback_dir = path.parent / "_rollback" / f"{path.stem}-{ts}"
    rollback_dir.mkdir(parents=True, exist_ok=True)
    for s in siblings:
        shutil.copy2(s, rollback_dir / s.name)
    return rollback_dir


def _quarantine_if_overshipped(
    saved_paths: list,
    train_acc: float,
    val_acc: float,
    instrument: str,
    model_name: str,
    restore_dir=None,
) -> dict:
    """Enforce the operator's hard 10% train-val gap ship rule.

    If gap > HARD_MAX_GAP, move every saved artifact (e.g. .keras + sidecar
    .meta.pkl + .ema.pkl + .ewc.pkl + .arch.json + .weights.h5) to a
    timestamped quarantine subdirectory so Tier 7 per-pair routing cannot
    pick it up. When ``restore_dir`` is provided (from
    ``_snapshot_existing_artifacts``), the prior known-good artifacts are
    restored into the live pair dir so routing keeps serving the last model
    that passed the gate instead of being left with no model. Returns
    ``{"quarantined": bool, "gap": float, "quarantine_dir": str|None}`` for
    the RESULT line.
    """
    gap = abs(train_acc - val_acc)
    if gap <= HARD_MAX_GAP:
        return {"quarantined": False, "gap": gap, "quarantine_dir": None}

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    qdir = (MODELS_DIR / instrument / "_quarantine" / f"{model_name}-{ts}")
    qdir.mkdir(parents=True, exist_ok=True)
    moved = []
    live_dir = None
    for p in saved_paths:
        path = Path(p)
        live_dir = path.parent
        # Match every sidecar that shares the file's stem in the pair dir.
        stem = path.stem
        for sibling in path.parent.glob(f"{stem}*"):
            if sibling.is_file():
                dest = qdir / sibling.name
                sibling.rename(dest)
                moved.append(str(dest))
    restored = 0
    if restore_dir is not None and live_dir is not None:
        rb = Path(restore_dir)
        if rb.is_dir():
            for f in rb.iterdir():
                if f.is_file():
                    shutil.copy2(f, live_dir / f.name)
                    restored += 1
    logger.error(
        f"🚫 SHIP-BLOCKED: {instrument}/{model_name} gap={gap:.4f} > {HARD_MAX_GAP} "
        f"(train={train_acc:.4f}, val={val_acc:.4f}). Quarantined {len(moved)} "
        f"file(s) → {qdir}; restored {restored} prior artifact(s) to live dir. "
        f"Operator rule: no models shipped > 10% gap."
    )
    return {"quarantined": True, "gap": gap, "quarantine_dir": str(qdir)}

# 2026-05-19 — close train/val embargo leak. The direction labeler uses
# `lookahead` bars of future return to assign y. Without `gap=lookahead` between
# train/val (and val/test), the first `lookahead` sequences of val are labeled
# from price moves that train sequences already saw — classic label leakage
# that silently inflates val_acc. Keep `lookahead` and `gap` lockstep so the
# embargo always matches the labeler horizon (no drift if defaults change).
# Source: docs/superpowers/plans/2026-05-19-val-acc-methodology-audit.md (#3).
DIRECTION_LOOKAHEAD = 24

# ── Pre-flight gate thresholds (see docs/superpowers/plans/2026-05-19-training-architecture-autonomy-audit.md) ──
# Directly addresses the 2026-05-13 "Bug B" disk-full corruption pattern observed twice.
MIN_FREE_GB = 2.0  # Minimum free disk space (GiB) required before training starts
OANDA_PROBE_CONNECT_TIMEOUT = 5.0  # seconds
OANDA_PROBE_READ_TIMEOUT = 10.0  # seconds

# ── Full wandb-optimized params per master pair ──────────────────────────────
# Restored from correlation_group_config.py — NO capping for M1 memory
MASTER_PARAMS = {
    "GBP_JPY": {
        "seq_len": 120, "batch_size": 128, "lr": 0.000508, "patience": 10,
        "dropout_rate": 0.236, "l2_reg": 0.000149,
        "tcn_nb_filters": 256, "tcn_kernel_size": 3,
        "tcn_dilations": [1, 2, 4, 8, 16, 32],
        "top_features": 150, "mixed_precision": True,
    },
    "EUR_GBP": {
        "seq_len": 90, "batch_size": 128, "lr": 0.000415, "patience": 25,
        "dropout_rate": 0.363, "l2_reg": 0.000098,
        "tcn_nb_filters": 256, "tcn_kernel_size": 5,
        "tcn_dilations": [1, 2, 4, 8, 16, 32],
        "top_features": 150, "mixed_precision": False,
    },
    "AUD_USD": {
        "seq_len": 90, "batch_size": 128, "lr": 0.000630, "patience": 20,
        "dropout_rate": 0.183, "l2_reg": 0.000405,
        "tcn_nb_filters": 64, "tcn_kernel_size": 5,
        "tcn_dilations": [1, 2, 4, 8, 16, 32],
        "top_features": None, "mixed_precision": True,
    },
    "USD_CAD": {
        # 2026-05-19 capacity-reduction experiment FAILED. Tried seq_len=60
        # (was 120), l2_reg=1e-3 (was 1.3e-5, ~80×), patience=10 (was 25).
        # Trainer's auto-helpers ALSO fired hard: Dropout +15% twice,
        # weight perturbation, momentum reset. Final result still gap=29.4%
        # (train=76.0%, val=46.6%, balanced=46.6%) — val BELOW chance.
        # See trained_data/logs/m15_usd_cad_regularization_2026_05_19.log.
        # Per CLAUDE.md modernization roadmap, USD_CAD M15 price-only is
        # shelved as unlearnable; news-fusion P1 is the next lever (design
        # at docs/superpowers/plans/2026-05-08-news-macro-signal-design.md).
        # Reverted to canonical params; further ad-hoc retrains will keep
        # producing quarantines until the news signal lands.
        "seq_len": 120, "batch_size": 128, "lr": 0.000552, "patience": 25,
        "dropout_rate": 0.468, "l2_reg": 0.000013,
        "tcn_nb_filters": 128, "tcn_kernel_size": 5,
        "tcn_dilations": [1, 2, 4, 8, 16],
        "top_features": 150, "mixed_precision": False,
    },
    "AUD_NZD": {
        "seq_len": 90, "batch_size": 128, "lr": 0.000376, "patience": 20,
        "dropout_rate": 0.420, "l2_reg": 0.000208,
        "tcn_nb_filters": 256, "tcn_kernel_size": 7,
        "tcn_dilations": [1, 2, 4, 8, 16],
        "top_features": 100, "mixed_precision": True,
    },
}

# Default for non-master / target pairs (uses EUR_GBP base)
DEFAULT_PARAMS = {
    "seq_len": 90, "batch_size": 128, "lr": 0.000415, "patience": 25,
    "dropout_rate": 0.363, "l2_reg": 0.000098,
    "tcn_nb_filters": 256, "tcn_kernel_size": 5,
    "tcn_dilations": [1, 2, 4, 8, 16, 32],
    "top_features": 150, "mixed_precision": False,
}


def check_metal_gpu():
    """Check and report tf-metal GPU availability."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"✓ Metal GPU detected: {gpus}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        else:
            logger.warning("⚠ No Metal GPU found — training will use CPU")
            return False
    except Exception as e:
        logger.warning(f"⚠ GPU check failed: {e}")
        return False


def fetch_or_load(instrument: str, candles: int, granularity: str) -> pd.DataFrame:
    """Fetch from OANDA or load cached CSV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{instrument}_{granularity}_{candles}.csv"

    if cache_file.exists():
        logger.info(f"Loading cached data: {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["time"])
        return df

    from src.utils.oanda_practice import OandaPracticeClient
    client = OandaPracticeClient.from_env()
    all_rows = []
    remaining = candles
    to_time = None

    logger.info(f"Fetching {candles:,} {granularity} candles for {instrument}...")
    while remaining > 0:
        batch_size = min(remaining, 5000)
        kwargs = {"granularity": granularity, "count": batch_size}
        if to_time:
            kwargs["to_time"] = to_time
        try:
            response = client.get_candles(instrument, **kwargs)
        except Exception as e:
            logger.warning(f"Fetch error (retrying): {e}")
            time.sleep(2)
            try:
                response = client.get_candles(instrument, **kwargs)
            except Exception as e2:
                logger.error(f"Fetch failed after retry: {e2}")
                break

        candles_list = response.get("candles", [])
        if not candles_list:
            break
        for c in candles_list:
            mid = c.get("mid", {})
            all_rows.append({
                "time": c.get("time"),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })
        remaining -= len(candles_list)
        to_time = candles_list[0].get("time")
        if len(candles_list) < batch_size:
            break
        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        df.to_csv(cache_file, index=False)

    logger.info(f"✓ {instrument}: {len(df):,} candles")
    return df


def apply_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering."""
    from src.utils import load_config
    from src.data.feature_engineering import FeatureEngineering
    cfg = load_config(CONFIG_PATH)
    fe = FeatureEngineering(cfg)
    return fe.create_features(df.copy(), include_all=True)


def gc_cleanup():
    """Force garbage collection + clear TF session."""
    gc.collect()
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def _git_short_sha() -> str:
    """Return short git SHA for the current HEAD, or 'unknown' on any failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = (out.stdout or "").strip()
        return sha or "unknown"
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(f"git rev-parse failed: {exc}")
        return "unknown"


def _write_status_json(
    instrument: str,
    granularity: str,
    candles: int,
    model_filter: str,
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    git_commit: str,
    results: list,
    errors: list,
) -> Path | None:
    """Atomically write per-run unified training_status.json for agent consumption.

    Schema v1 — see docs/superpowers/plans/2026-05-19-training-architecture-autonomy-audit.md
    recommendation #1. One file per pair, overwritten each retrain. Failures here MUST
    NOT crash the training run — log a WARNING and return None.
    """
    try:
        passed_count = sum(1 for r in results if r.get("passed"))
        failed_count = sum(1 for r in results if not r.get("passed"))
        quarantined_count = sum(1 for r in results if r.get("quarantined"))

        payload = {
            "schema_version": 1,
            "instrument": instrument,
            "granularity": granularity,
            "candles": candles,
            "model_filter": model_filter,
            "started_at_utc": started_at_utc,
            "completed_at_utc": completed_at_utc,
            "duration_s": duration_s,
            "git_commit": git_commit,
            "results": results,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "quarantined_count": quarantined_count,
            "errors": errors,
        }

        out_dir = MODELS_DIR / instrument
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / "training_status.json"
        tmp_path = out_dir / "training_status.json.tmp"

        serialized = json.dumps(payload, indent=2, sort_keys=True, default=str)
        with open(tmp_path, "w") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, final_path)
        logger.info(f"✓ Wrote {final_path}")
        return final_path
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(f"_write_status_json failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINERS — one function per model type
# ═══════════════════════════════════════════════════════════════════════════════

def train_transformer(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """Transformer Direction Trainer — primary direction prediction model."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.transformer_trainer import TransformerDirectionTrainer
    from src.core.modular_data_loaders import load_direction_data

    cfg = TrainerConfig(
        epochs=200, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=params["seq_len"], min_epochs=30,
        label_smoothing=0.1, use_augmentation=True, use_class_weights=True,
        use_ema=True, use_ewc=False, use_feature_selection=True,
        use_replay_buffer=False, overfit_threshold=0.04, critical_threshold=0.08,
        max_acceptable_gap=MAX_GAP,
    )

    dir_data = load_direction_data(df_feat, lookahead=DIRECTION_LOOKAHEAD, gap=DIRECTION_LOOKAHEAD)
    if not dir_data:
        return {"error": "No direction data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = TransformerDirectionTrainer(cfg)
    # 2026-05-13 fix: forward w_train/w_val from load_direction_data so the
    # trainer's `w_train_seq > 0` clear-label filter actually fires. Without
    # these, create_sequences_with_weights substitutes uniform 1.0 weights
    # (utils.py:304-307) → 100% of sequences pass the "clear" filter → 60%
    # unclear (y=0.5) samples leak into Keras binary_accuracy as SHORT and
    # drag val_accuracy to ~17% with balanced_acc=50% (chance).
    trainer.train(dir_data["X_train"], dir_data["y_train"],
                  dir_data["X_val"], dir_data["y_val"],
                  w_train=dir_data.get("w_train"),
                  w_val=dir_data.get("w_val"),
                  feature_names=dir_data.get("feature_names"))

    save_path = save_dir / "transformer_direction.keras"
    trainer.save(str(save_path))

    metrics = _read_trainer_metrics(trainer)
    train_acc = float(metrics.get("train_accuracy", 0.0))
    val_acc = float(metrics.get("val_accuracy", 0.0))
    ship = _quarantine_if_overshipped(
        [save_path], train_acc, val_acc, instrument, "transformer_direction"
    )
    gc_cleanup()
    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "gap": ship["gap"],
        "quarantined": ship["quarantined"],
        "quarantine_dir": ship["quarantine_dir"],
    }


def train_tcn(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """TCN Volatility Regime Trainer — 4-class volatility classification."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.tcn_trainer import TCNTrainer
    from src.core.modular_data_loaders import load_volatility_regime_data

    cfg = TrainerConfig(
        epochs=200, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=params["seq_len"], min_epochs=30,
        use_replay_buffer=False, overfit_threshold=0.04, max_acceptable_gap=MAX_GAP,
    )

    tcn_data = load_volatility_regime_data(df_feat, instrument=instrument)
    if not tcn_data:
        return {"error": "No volatility regime data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "tcn_volatility_regime.keras"
    snapshot_dir = _snapshot_existing_artifacts(save_path)

    trainer = TCNTrainer(cfg)
    trainer.train(tcn_data["X_train"], tcn_data["y_train"],
                  tcn_data["X_val"], tcn_data["y_val"])
    trainer.save(str(save_path))

    # "tcn" is GAP_CHECKED: the legacy get_metrics() reader silently returned {}
    # (no trainer defines it) -> gap=0 -> the 10%/6% gates were permanent no-ops.
    metrics = _read_trainer_metrics(trainer)
    train_acc = float(metrics.get("train_accuracy", 0) or 0)
    val_acc = float(metrics.get("val_accuracy", 0) or 0)
    q = _quarantine_if_overshipped(
        [save_path], train_acc=train_acc, val_acc=val_acc,
        instrument=instrument, model_name="tcn_volatility_regime",
        restore_dir=snapshot_dir,
    )
    gc_cleanup()
    return {
        "train_acc": train_acc, "val_acc": val_acc, "gap": q["gap"],
        "quarantined": q["quarantined"], "quarantine_dir": q.get("quarantine_dir"),
    }


def train_lgbm_momentum(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """LightGBM Momentum Trainer — momentum signal scoring."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.lightgbm_trainers import LightGBMMomentumTrainer
    from src.core.modular_data_loaders import load_xgboost_data

    cfg = TrainerConfig(batch_size=params["batch_size"], learning_rate=params["lr"])
    data = load_xgboost_data(df_feat)
    if not data:
        return {"error": "No XGBoost data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = LightGBMMomentumTrainer(cfg)
    trainer.train(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    trainer.save(str(save_dir / "lgbm_momentum.pkl"))

    metrics = _read_trainer_metrics(trainer)
    gc_cleanup()
    return {"train_acc": metrics.get("train_accuracy", 0), "val_acc": metrics.get("val_accuracy", 0)}


def train_lgbm_risk(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """LightGBM Risk Trainer — drawdown risk assessment."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.lightgbm_trainers import LightGBMRiskTrainer
    from src.core.modular_data_loaders import load_rf_data

    cfg = TrainerConfig(batch_size=params["batch_size"], learning_rate=params["lr"])
    data = load_rf_data(df_feat)
    if not data:
        return {"error": "No RF data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = LightGBMRiskTrainer(cfg)
    trainer.train(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    trainer.save(str(save_dir / "lgbm_risk.pkl"))

    metrics = _read_trainer_metrics(trainer)
    gc_cleanup()
    return {"train_acc": metrics.get("train_accuracy", 0), "val_acc": metrics.get("val_accuracy", 0)}


def train_ridge(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """Ridge/LightGBM Confidence Trainer — confidence/stability scoring."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.ridge_trainer import RidgeTrainer
    from src.core.modular_data_loaders import load_ridge_data

    cfg = TrainerConfig(batch_size=params["batch_size"], learning_rate=params["lr"])
    data = load_ridge_data(df_feat)
    if not data:
        return {"error": "No Ridge data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = RidgeTrainer(cfg)
    trainer.train(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
    trainer.save(str(save_dir / "ridge_confidence.pkl"))

    metrics = _read_trainer_metrics(trainer)
    gc_cleanup()
    return {"train_acc": metrics.get("train_accuracy", 0), "val_acc": metrics.get("val_accuracy", 0)}


def train_histgb(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """HistGradientBoosting Direction Trainer — fast gradient boosting for direction."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.histgb_trainer import HistGradientBoostingDirectionTrainer
    from src.core.modular_data_loaders import load_direction_data

    cfg = TrainerConfig(batch_size=params["batch_size"], learning_rate=params["lr"])
    dir_data = load_direction_data(df_feat, lookahead=DIRECTION_LOOKAHEAD, gap=DIRECTION_LOOKAHEAD)
    if not dir_data:
        return {"error": "No direction data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train = dir_data["X_train"]
    X_val = dir_data["X_val"]
    if len(X_train.shape) == 3:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)

    trainer = HistGradientBoostingDirectionTrainer(cfg)
    # 2026-05-13 fix: forward w_train/w_val so the HistGB trainer can drop
    # unclear (weight=0) samples. Without these the trainer just predicts
    # the SHORT majority class → val_accuracy=82%/balanced_acc=50% (the
    # majority-class-only fingerprint observed in the 2026-05-12 retrain).
    trainer.train(X_train, dir_data["y_train"], X_val, dir_data["y_val"],
                  w_train=dir_data.get("w_train"),
                  w_val=dir_data.get("w_val"))

    save_path = save_dir / "histgb_direction.pkl"
    trainer.save(str(save_path))

    metrics = _read_trainer_metrics(trainer)
    train_acc = float(metrics.get("train_accuracy", 0.0))
    val_acc = float(metrics.get("val_accuracy", 0.0))
    ship = _quarantine_if_overshipped(
        [save_path], train_acc, val_acc, instrument, "histgb_direction"
    )
    gc_cleanup()
    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "gap": ship["gap"],
        "quarantined": ship["quarantined"],
        "quarantine_dir": ship["quarantine_dir"],
    }


def train_transformer_regime(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """Transformer Regime Trainer — market regime classification via transformer."""
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers.transformer_regime_trainer import TransformerRegimeTrainer
    from src.core.modular_data_loaders import load_direction_data

    cfg = TrainerConfig(
        epochs=200, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=params["seq_len"], min_epochs=30,
        use_replay_buffer=False, overfit_threshold=0.04, max_acceptable_gap=MAX_GAP,
    )

    dir_data = load_direction_data(df_feat, lookahead=DIRECTION_LOOKAHEAD, gap=DIRECTION_LOOKAHEAD)
    if not dir_data:
        return {"error": "No direction data"}

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = TransformerRegimeTrainer(cfg)
    trainer.train(dir_data["X_train"], dir_data["y_train"],
                  dir_data["X_val"], dir_data["y_val"])
    trainer.save(str(save_dir / "transformer_regime.keras"))

    metrics = _read_trainer_metrics(trainer)
    gc_cleanup()
    return {"val_acc": metrics.get("val_accuracy", 0)}


def train_tcn_vol_regime(instrument: str, df_feat: pd.DataFrame, params: dict) -> dict:
    """Forward-volatility TCN regime head (dual-head: classification + regression).

    Trains the 4-class volatility-regime classifier alongside a forward-vol
    %-change regression head. The regression output + ``reg_thresholds`` are the
    inference contract the gate's ``_normalize_tcn_volatility_output`` reads
    (commit b9bec7e) to recover the regime when classification confidence is
    low. The forward-sequence loader returns 3D ``(batch, seq_len, features)``
    arrays already, plus the regression targets, sample/class weights, and a
    fitted scaler — so this function only wires the loader → trainer → save →
    gap-gate path. Modules (not names) are imported so tests can monkeypatch
    the loader and trainer.
    """
    from src.training.trainers.config import TrainerConfig
    from src.training.trainers import tcn_volatility_trainer as trainer_mod
    from src.core import modular_data_loaders as loaders

    seq_len = params["seq_len"]
    dilations = params.get("tcn_dilations", [1, 2, 4, 8, 16])

    vol_data = loaders.load_forward_volatility_data(df_feat, seq_len=seq_len)
    if not vol_data:
        return {"error": "No forward volatility data"}

    cfg = TrainerConfig(
        epochs=200, batch_size=params["batch_size"], learning_rate=params["lr"],
        patience=params["patience"], seq_len=seq_len, min_epochs=30,
        use_replay_buffer=False, overfit_threshold=0.04, max_acceptable_gap=MAX_GAP,
    )
    # TrainerConfig has tcn_kernel_size as a field but not the residual-block
    # count or filter width; the trainer reads both via getattr. Set them as
    # attributes (non-frozen dataclass). Residual blocks = number of dilations.
    cfg.tcn_kernel_size = params["tcn_kernel_size"]
    cfg.tcn_num_residual_blocks = len(dilations)

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "tcn_volatility_regime.keras"

    # Snapshot the prior live model BEFORE overwrite so a gap-gate quarantine
    # can roll back to the last known-good artifacts instead of leaving the
    # pair with no volatility-regime model.
    snapshot_dir = _snapshot_existing_artifacts(save_path)

    trainer = trainer_mod.TCNVolatilityRegimeTrainer(cfg)
    trainer.train(
        vol_data["X_train"], vol_data["y_train"],
        vol_data["X_val"], vol_data["y_val"],
        feature_names=vol_data.get("feature_names"),
        class_weights=vol_data.get("class_weights"),
        seq_len=seq_len,
        y_train_reg=vol_data.get("y_train_reg"),
        y_val_reg=vol_data.get("y_val_reg"),
        w_train=vol_data.get("w_train"),
        w_val=vol_data.get("w_val"),
    )

    # Attach the inference contract bits the gate reads back from the model.
    if vol_data.get("reg_thresholds"):
        trainer.reg_thresholds = vol_data["reg_thresholds"]
    if vol_data.get("lookahead") is not None:
        trainer.lookahead = vol_data["lookahead"]
    if vol_data.get("scaler") is not None:
        trainer.scaler = vol_data["scaler"]
    trainer.save(str(save_path))

    metrics = getattr(trainer, "metrics", {}) or {}
    train_acc = float(metrics.get("train_accuracy", 0.0) or 0.0)
    val_acc = float(metrics.get("val_accuracy", 0.0) or 0.0)

    q = _quarantine_if_overshipped(
        [save_path], train_acc=train_acc, val_acc=val_acc,
        instrument=instrument, model_name="tcn_volatility_regime",
        restore_dir=snapshot_dir,
    )
    gc_cleanup()
    return {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "gap": q["gap"],
        "quarantined": q["quarantined"],
        "quarantine_dir": q.get("quarantine_dir"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_TRAINERS = {
    "transformer": train_transformer,
    "tcn": train_tcn,
    "lgbm_momentum": train_lgbm_momentum,
    "lgbm_risk": train_lgbm_risk,
    "ridge": train_ridge,
    "histgb": train_histgb,
    # transformer_regime removed — TCN is the sole regime model (65.8% acc vs broken transformer_regime)
    "tcn_vol_regime": train_tcn_vol_regime,
}

# Direction-critical models that get gap-checked
GAP_CHECKED_MODELS = {"transformer", "tcn", "histgb"}

# Models that `--model all` must NOT train in the standard per-pair sweep.
# tcn_vol_regime is a forward-volatility regression head trained explicitly
# on demand (it has its own loader + dual-head trainer), not part of the
# default direction/momentum/risk batch.
_MODEL_ALL_SKIP = {"tcn_vol_regime"}


def _model_names_for_request(model: str) -> list:
    """Expand a ``--model`` request into the concrete trainer names to run.

    ``"all"`` → every registered trainer except the aliases in
    ``_MODEL_ALL_SKIP``. Any specific name passes through unchanged (so
    ``tcn_vol_regime`` can still be trained explicitly).
    """
    if model == "all":
        return [m for m in MODEL_TRAINERS if m not in _MODEL_ALL_SKIP]
    return [model]


def _preflight_check(args) -> None:
    """Pre-flight gate. HARD STOP before any training begins.

    Validates the two failure modes that caused the 2026-05-13 "Bug B" disk-full
    corruption (val_acc=0.4839 model shipped via 2/3-pass deploy gate):
      1. Disk free space ≥ MIN_FREE_GB on PROJECT_ROOT volume.
      2. OANDA connectivity probe (short timeout, no retries — fail fast).

    Exit code 2 on failure (distinguishes from training failures which exit 1).
    Bypassable via --skip-preflight for dev / local debugging.
    """
    if getattr(args, "skip_preflight", False):
        logger.warning("⚠ pre-flight skipped via --skip-preflight (dev mode)")
        return

    # 1. Disk free space check
    try:
        usage = shutil.disk_usage(str(PROJECT_ROOT))
        free_gb = usage.free / (1024 ** 3)
    except OSError as e:
        logger.error(f"✗ pre-flight FAIL: disk_usage({PROJECT_ROOT}) raised {type(e).__name__}: {e}")
        logger.error("  remediation: verify PROJECT_ROOT path exists and is readable")
        raise SystemExit(2)

    if free_gb < MIN_FREE_GB:
        logger.error(
            f"✗ pre-flight FAIL: disk free {free_gb:.2f} GiB < required {MIN_FREE_GB} GiB on {PROJECT_ROOT}"
        )
        logger.error(
            "  remediation: free up space in trained_data/ (cache, replay, old models) or pass --skip-preflight"
        )
        raise SystemExit(2)
    logger.info(f"✓ pre-flight: disk free {free_gb:.2f} GiB (≥ {MIN_FREE_GB} GiB required)")

    # 2. OANDA connectivity probe (no retries — fail fast)
    import requests

    api_token = os.getenv("OANDA_API_TOKEN") or os.getenv("OANDA_API_KEY")
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    if not api_token:
        logger.error("✗ pre-flight FAIL: OANDA_API_TOKEN / OANDA_API_KEY not set in env")
        logger.error("  remediation: source .env.local or set OANDA_API_KEY, or pass --skip-preflight")
        raise SystemExit(2)
    if not account_id:
        logger.error("✗ pre-flight FAIL: OANDA_ACCOUNT_ID not set in env")
        logger.error("  remediation: source .env.local or set OANDA_ACCOUNT_ID, or pass --skip-preflight")
        raise SystemExit(2)

    probe_url = "https://api-fxpractice.oanda.com/v3/accounts"
    try:
        resp = requests.get(
            probe_url,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=(OANDA_PROBE_CONNECT_TIMEOUT, OANDA_PROBE_READ_TIMEOUT),
        )
    except requests.Timeout as e:
        logger.error(f"✗ pre-flight FAIL: OANDA probe timed out: {e}")
        logger.error(
            f"  remediation: check network; probe used connect={OANDA_PROBE_CONNECT_TIMEOUT}s read={OANDA_PROBE_READ_TIMEOUT}s"
        )
        raise SystemExit(2)
    except requests.ConnectionError as e:
        logger.error(f"✗ pre-flight FAIL: OANDA probe connection error: {e}")
        logger.error("  remediation: check DNS/firewall; api-fxpractice.oanda.com must be reachable")
        raise SystemExit(2)
    except requests.RequestException as e:
        logger.error(f"✗ pre-flight FAIL: OANDA probe failed: {type(e).__name__}: {e}")
        raise SystemExit(2)

    if resp.status_code >= 400:
        logger.error(
            f"✗ pre-flight FAIL: OANDA probe HTTP {resp.status_code} from {probe_url}"
        )
        body_preview = resp.text[:200] if resp.text else "<empty>"
        logger.error(f"  body: {body_preview}")
        logger.error("  remediation: verify OANDA_API_KEY is valid and not expired")
        raise SystemExit(2)
    logger.info(f"✓ pre-flight: OANDA reachable (HTTP {resp.status_code})")
    logger.info("✓ pre-flight passed; proceeding to training")


def main():
    parser = argparse.ArgumentParser(description="Train single model (M1 Mac optimized)")
    parser.add_argument("--instrument", required=True, help="e.g. GBP_JPY")
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_TRAINERS.keys()) + ["all"],
                        help="Model type or 'all' for all 8 models")
    parser.add_argument("--candles", type=int, default=25000, help="Number of H1 candles")
    parser.add_argument("--granularity", default="H1")
    parser.add_argument("--no-gpu-check", action="store_true", help="Skip Metal GPU check")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pre-flight disk/OANDA gate (dev / local debugging only)",
    )
    args = parser.parse_args()

    # Reproducibility: seed python/np/tf BEFORE any graph construction so two
    # retrains of identical data ship identical models (the 10% gap-gate at
    # the boundary must not be an RNG coin-flip). Seed recorded in
    # training_status.json via _write_status_json.
    from src.utils.seed_manager import set_global_seed
    set_global_seed(int(os.environ.get("BUDDY_TRAIN_SEED", "42")))

    # Pre-flight gate — HARD STOP if disk low or OANDA unreachable (exit 2)
    _preflight_check(args)

    # GPU check
    if not args.no_gpu_check:
        check_metal_gpu()

    t0 = time.time()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    git_commit = _git_short_sha()
    top_level_errors: list = []
    instrument = args.instrument
    params = MASTER_PARAMS.get(instrument, DEFAULT_PARAMS.copy())

    # Fetch/load data
    df = fetch_or_load(instrument, args.candles, args.granularity)
    if df.empty:
        print(json.dumps({"error": "No data", "instrument": instrument, "model": args.model}))
        top_level_errors.append({"stage": "fetch_or_load", "error": "No data"})
        _write_status_json(
            instrument=instrument,
            granularity=args.granularity,
            candles=args.candles,
            model_filter=args.model,
            started_at_utc=started_at_utc,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            duration_s=round(time.time() - t0, 1),
            git_commit=git_commit,
            results=[],
            errors=top_level_errors,
        )
        sys.exit(1)

    # Feature engineering
    df_feat = apply_features(df)
    del df
    gc.collect()

    models_to_train = _model_names_for_request(args.model)
    results = []

    for model_name in models_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name} for {instrument} ({len(df_feat)} rows, {len(df_feat.columns)} features)")
        logger.info(f"{'='*60}")

        trainer_fn = MODEL_TRAINERS[model_name]
        try:
            result = trainer_fn(instrument, df_feat, params)
            result["instrument"] = instrument
            result["model"] = model_name
            result["duration_s"] = round(time.time() - t0, 1)

            gap = result.get("gap", 0)
            # Hard ship rule (operator 2026-05-13): gap > 10% → quarantined.
            # If the trainer already moved files to _quarantine/, propagate
            # the BLOCK status regardless of which model_name; the cosmetic
            # PASS/FAIL still uses MAX_GAP=6% for the gap-checked subset.
            if result.get("quarantined"):
                result["passed"] = False
                logger.info(
                    f"🚫 SHIP-BLOCKED: {model_name} gap={gap:.4f} > "
                    f"{HARD_MAX_GAP} (quarantined → {result.get('quarantine_dir')})"
                )
            elif model_name in GAP_CHECKED_MODELS:
                # Fail loud on the legacy gap=0 liar: a gap-checked model whose
                # metric-read path returned nothing must NOT ship green
                # (improvement.md Hard Ship Gate: "gap=0 pattern means the
                # metric-read path is broken — fail loud").
                if not result.get("train_acc") and not result.get("val_acc"):
                    result["passed"] = False
                    result["error"] = "empty metrics for gap-checked model (metric-read path broken)"
                    logger.error(
                        f"✗ METRICS-EMPTY: {model_name} reported train_acc=val_acc=0; "
                        f"refusing PASS — fix the trainer's self.metrics population."
                    )
                else:
                    result["passed"] = gap <= MAX_GAP
                    status = "✓ PASS" if result["passed"] else "✗ FAIL"
                    logger.info(f"{status}: {model_name} gap={gap:.4f} (max={MAX_GAP})")
            else:
                result["passed"] = True
                logger.info(f"✓ DONE: {model_name} (no gap check)")

            results.append(result)
            print(f"RESULT:{json.dumps(result)}")
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            logger.error(traceback.format_exc())
            error_result = {
                "error": str(e), "instrument": instrument, "model": model_name,
                "duration_s": round(time.time() - t0, 1), "passed": False,
            }
            results.append(error_result)
            print(f"RESULT:{json.dumps(error_result)}")

        gc_cleanup()

    # Summary
    total_time = round(time.time() - t0, 1)
    passed = sum(1 for r in results if r.get("passed"))
    failed = sum(1 for r in results if not r.get("passed"))
    errors = sum(1 for r in results if "error" in r)

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE: {instrument}")
    logger.info(f"  Models: {len(results)} total, {passed} passed, {failed} failed, {errors} errors")
    logger.info(f"  Duration: {total_time}s")
    logger.info(f"{'='*60}")

    # Write summary to results file
    summary_path = MODELS_DIR / instrument / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "instrument": instrument,
            "candles": args.candles,
            "params": params,
            "results": results,
            "total_duration_s": total_time,
            "passed": passed,
            "failed": failed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2, sort_keys=True, default=str)

    # Write unified per-run training_status.json (schema v1) for autonomous agents.
    # Per docs/superpowers/plans/2026-05-19-training-architecture-autonomy-audit.md.
    _write_status_json(
        instrument=instrument,
        granularity=args.granularity,
        candles=args.candles,
        model_filter=args.model,
        started_at_utc=started_at_utc,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        duration_s=total_time,
        git_commit=git_commit,
        results=results,
        errors=top_level_errors,
    )

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
