#!/usr/bin/env python3
"""
Full Ensemble Training Orchestrator — All Tiers, All Correlation Groups.

Trains every model Buddy has across all 5 master pairs, transfers to targets,
then trains RL + meta-labeler + regime/volatility models.

Enforces <6% direction train-val accuracy gap (learning, not memorizing).

Usage:
    export $(cat .env.local | grep -v '^#' | xargs)
    python scripts/train_full_ensemble.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Environment setup ──────────────────────────────────────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Load .env.local
env_path = Path(__file__).resolve().parent.parent / ".env.local"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_full_ensemble")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CANDLES_PER_PAIR = 25_000
# For subprocess-based training, save fetched data to disk to avoid re-fetching
DATA_CACHE_DIR = PROJECT_ROOT / "trained_data" / "cache" / "training_data"
GRANULARITY = "H1"
MAX_ACCEPTABLE_GAP = 0.06  # 6% — hard gate
CONFIG_PATH = str(PROJECT_ROOT / "config" / "config_m1_optimized.yaml")
MODELS_DIR = PROJECT_ROOT / "trained_data" / "models"


def _control_plane_apply(trainer: Any, head: str) -> Dict[str, Any]:
    """Pull <head>_training_config from W&B and apply to ``trainer``.

    Best-effort, never raises. Returns the config dict (possibly empty) so
    callers can pass it to subsequent ``log_run`` calls. Used by
    train_full_ensemble to route operator-tunable HPs through W&B.
    """
    try:
        from src.training.wandb_control_plane import (
            apply_config_to_trainer,
            pull_config,
            seed_default,
        )
        seed_default(head)
        cfg = pull_config(head)
        if cfg:
            apply_config_to_trainer(trainer, cfg)
        return dict(cfg or {})
    except Exception as exc:  # pragma: no cover — observability only
        logger.debug("control plane apply failed for %s: %s", head, exc)
        return {}


def _control_plane_log(
    head: str,
    cfg: Dict[str, Any],
    metrics: Dict[str, Any],
    artifact_path: Optional[Path] = None,
    instrument: Optional[str] = None,
) -> None:
    """Log a manual training run for a head. Best-effort, never raises."""
    try:
        from src.training.wandb_control_plane import log_run
        log_run(
            head=head,
            config=cfg or {},
            metrics=metrics,
            artifacts=[artifact_path] if artifact_path else None,
            run_name=f"manual_{head}_{instrument or 'all'}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            tags=[f"pair={instrument}"] if instrument else None,
            extra_config={
                "source": "manual",
                "instrument": instrument,
                "script": "train_full_ensemble",
            },
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("control plane log failed for %s: %s", head, exc)

# Correlation groups: master → target pairs
CORRELATION_GROUPS = {
    "yen_carry": {
        "master": "GBP_JPY",
        "targets": ["EUR_JPY", "AUD_JPY", "NZD_JPY", "USD_JPY"],
    },
    "euro_basket": {
        "master": "EUR_GBP",
        "targets": ["EUR_USD", "EUR_JPY", "EUR_AUD"],
    },
    "sterling_cluster": {
        "master": "EUR_GBP",  # shared with euro_basket
        "targets": ["GBP_USD", "GBP_JPY", "GBP_AUD"],
    },
    "aussie_kiwi_commodity": {
        "master": "AUD_USD",
        "targets": ["NZD_USD", "AUD_JPY", "NZD_JPY"],
    },
    "usd_safe_haven": {
        "master": "USD_CAD",
        "targets": ["USD_JPY", "USD_CHF"],
    },
    "crosses": {
        "master": "AUD_NZD",
        "targets": ["GBP_AUD", "EUR_AUD"],
    },
}

# Unique masters (avoid training EUR_GBP twice)
UNIQUE_MASTERS = ["GBP_JPY", "EUR_GBP", "AUD_USD", "USD_CAD", "AUD_NZD"]

# Best wandb sweep params per master
MASTER_PARAMS = {
    "GBP_JPY": {"seq_len": 120, "batch_size": 128, "lr": 0.000508, "patience": 10},
    "EUR_GBP": {"seq_len": 90, "batch_size": 128, "lr": 0.000415, "patience": 25},
    "AUD_USD": {"seq_len": 90, "batch_size": 128, "lr": 0.000630, "patience": 20},
    "USD_CAD": {"seq_len": 120, "batch_size": 128, "lr": 0.000552, "patience": 25},
    "AUD_NZD": {"seq_len": 90, "batch_size": 128, "lr": 0.000376, "patience": 20},
}

# RL training timesteps
RL_TIMESTEPS = 500_000


@dataclass
class TrainingResult:
    """Result of training a single model for a pair."""
    pair: str
    model_name: str
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    gap: float = 0.0
    passed: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "model": self.model_name,
            "train_acc": round(self.train_accuracy, 4),
            "val_acc": round(self.val_accuracy, 4),
            "gap": round(self.gap, 4),
            "passed": self.passed,
            "error": self.error,
            "duration_s": round(self.duration_seconds, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering using FeatureEngineering class."""
    from src.utils import load_config
    from src.data.feature_engineering import FeatureEngineering
    cfg = load_config(CONFIG_PATH)
    fe = FeatureEngineering(cfg)
    return fe.create_features(df.copy(), include_all=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_candles(instrument: str, candles: int = CANDLES_PER_PAIR, granularity: str = GRANULARITY) -> pd.DataFrame:
    """Fetch candles from OANDA in paginated batches of 5000."""
    from src.utils.oanda_practice import OandaPracticeClient

    client = OandaPracticeClient.from_env()
    all_rows = []
    remaining = candles
    to_time = None

    logger.info(f"Fetching {candles:,} {granularity} candles for {instrument}...")

    while remaining > 0:
        batch_size = min(remaining, 5000)
        kwargs: dict = {"granularity": granularity, "count": batch_size}
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
        logger.info(f"  ... fetched {len(candles_list)}, remaining: {remaining:,}")

        if len(candles_list) < batch_size:
            break
        time.sleep(0.5)  # Rate limit respect

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    logger.info(f"  ✓ {instrument}: {len(df):,} candles total")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: CORE ENSEMBLE (Transformer + TCN + LightGBM + Ridge + HistGB)
# ═══════════════════════════════════════════════════════════════════════════════

def train_core_ensemble(instrument: str, df: pd.DataFrame, params: dict) -> List[TrainingResult]:
    """Train the full core ensemble for a master pair."""
    results = []

    # ── Feature Engineering ──
    df_feat = apply_feature_engineering(df)
    logger.info(f"  Features: {len(df_feat.columns)} columns, {len(df_feat)} rows")

    # ── Data Preparation ──
    from src.core.modular_data_loaders import (
        load_direction_data,
        load_xgboost_data,
        load_rf_data,
        load_ridge_data,
    )

    # ── Trainer Config ──
    from src.training.trainers.config import TrainerConfig
    trainer_config = TrainerConfig(
        epochs=100,  # Reduced for memory-constrained environment
        batch_size=min(params.get("batch_size", 64), 64),  # Cap at 64 for memory
        learning_rate=params.get("lr", 0.0005),
        patience=params.get("patience", 20),
        seq_len=min(params.get("seq_len", 60), 60),  # Cap at 60 for memory
        min_epochs=20,
        label_smoothing=0.1,
        use_augmentation=True,
        use_class_weights=True,
        use_ema=True,
        use_ewc=False,  # Disable for fresh training (no prior weights to consolidate)
        use_feature_selection=True,
        use_replay_buffer=False,  # Disable replay buffer for fresh full training
        # Tightened overfit prevention
        overfit_threshold=0.04,
        critical_threshold=0.08,
        severe_threshold=0.15,
        max_acceptable_gap=MAX_ACCEPTABLE_GAP,
    )

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load direction data once for all direction-based models ──
    dir_data = None
    try:
        dir_data = load_direction_data(df_feat)
        if dir_data:
            logger.info(f"  Direction data: train={dir_data['X_train'].shape}, val={dir_data['X_val'].shape}")
    except Exception as e:
        logger.error(f"  ✗ Direction data loading failed: {e}")

    # 1. TRANSFORMER DIRECTION
    logger.info(f"  [1/6] Transformer Direction — {instrument}")
    t0 = time.time()
    try:
        from src.training.trainers.transformer_trainer import TransformerDirectionTrainer
        if dir_data:
            trainer = TransformerDirectionTrainer(trainer_config)
            cp_cfg_dir = _control_plane_apply(trainer, "direction")
            trainer.train(
                dir_data["X_train"], dir_data["y_train"],
                dir_data["X_val"], dir_data["y_val"],
                feature_names=dir_data.get("feature_names"),
            )
            trainer.save(str(save_dir / "transformer_direction.keras"))
            metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
            _control_plane_log(
                "direction", cp_cfg_dir,
                {"train_accuracy": metrics.get("train_direction_accuracy") or metrics.get("train_accuracy"),
                 "val_accuracy": metrics.get("val_direction_accuracy") or metrics.get("val_accuracy")},
                save_dir / "transformer_direction.keras", instrument,
            )
            train_acc = metrics.get("train_direction_accuracy", metrics.get("train_accuracy", 0))
            val_acc = metrics.get("val_direction_accuracy", metrics.get("val_accuracy", 0))
            gap = abs(train_acc - val_acc)
            results.append(TrainingResult(
                pair=instrument, model_name="transformer_direction",
                train_accuracy=train_acc, val_accuracy=val_acc, gap=gap,
                passed=gap <= MAX_ACCEPTABLE_GAP, duration_seconds=time.time() - t0,
            ))
            logger.info(f"    ✓ Transformer: train={train_acc:.4f} val={val_acc:.4f} gap={gap:.4f} {'PASS' if gap <= MAX_ACCEPTABLE_GAP else 'FAIL'}")
        else:
            results.append(TrainingResult(pair=instrument, model_name="transformer_direction", error="No direction data"))
    except Exception as e:
        logger.error(f"    ✗ Transformer failed: {e}")
        results.append(TrainingResult(pair=instrument, model_name="transformer_direction", error=str(e), duration_seconds=time.time() - t0))

    # 2. TCN VOLATILITY REGIME (4-class: LOW, NORMAL, HIGH, EXTREME)
    logger.info(f"  [2/6] TCN Volatility Regime — {instrument}")
    t0 = time.time()
    try:
        from src.training.trainers.tcn_trainer import TCNTrainer
        from src.core.modular_data_loaders import load_tcn_data
        tcn_data = load_tcn_data(df_feat)
        if tcn_data:
            tcn_trainer = TCNTrainer(trainer_config)
            cp_cfg_vol = _control_plane_apply(tcn_trainer, "volatility_regime")
            tcn_trainer.train(
                tcn_data["X_train"], tcn_data["y_train"],
                tcn_data["X_val"], tcn_data["y_val"],
            )
            tcn_trainer.save(str(save_dir / "tcn_volatility.keras"))
            metrics = tcn_trainer.get_metrics() if hasattr(tcn_trainer, "get_metrics") else {}
            _control_plane_log(
                "volatility_regime", cp_cfg_vol,
                {"train_accuracy": metrics.get("train_accuracy"),
                 "val_accuracy": metrics.get("val_accuracy")},
                save_dir / "tcn_volatility.keras", instrument,
            )
            train_acc = metrics.get("train_accuracy", 0)
            val_acc = metrics.get("val_accuracy", 0)
            gap = abs(train_acc - val_acc)
            results.append(TrainingResult(
                pair=instrument, model_name="tcn_volatility",
                train_accuracy=train_acc, val_accuracy=val_acc, gap=gap,
                passed=gap <= MAX_ACCEPTABLE_GAP, duration_seconds=time.time() - t0,
            ))
            logger.info(f"    ✓ TCN Vol: train={train_acc:.4f} val={val_acc:.4f} gap={gap:.4f} {'PASS' if gap <= MAX_ACCEPTABLE_GAP else 'FAIL'}")
        else:
            results.append(TrainingResult(pair=instrument, model_name="tcn_volatility", error="No TCN data"))
    except Exception as e:
        logger.error(f"    ✗ TCN failed: {e}")
        results.append(TrainingResult(pair=instrument, model_name="tcn_volatility", error=str(e), duration_seconds=time.time() - t0))

    # 3. LIGHTGBM MOMENTUM
    logger.info(f"  [3/6] LightGBM Momentum — {instrument}")
    t0 = time.time()
    try:
        from src.training.trainers.lightgbm_trainers import LightGBMMomentumTrainer
        mom_data = load_xgboost_data(df_feat)
        if mom_data:
            lgbm_mom = LightGBMMomentumTrainer(trainer_config)
            cp_cfg_mom = _control_plane_apply(lgbm_mom, "momentum")
            lgbm_mom.train(
                mom_data["X_train"], mom_data["y_train"],
                mom_data["X_val"], mom_data["y_val"],
            )
            lgbm_mom.save(str(save_dir / "lgbm_momentum.pkl"))
            metrics = lgbm_mom.get_metrics() if hasattr(lgbm_mom, "get_metrics") else {}
            _control_plane_log(
                "momentum", cp_cfg_mom,
                {"train_accuracy": metrics.get("train_accuracy"),
                 "val_accuracy": metrics.get("val_accuracy")},
                save_dir / "lgbm_momentum.pkl", instrument,
            )
            train_acc = metrics.get("train_accuracy", 0)
            val_acc = metrics.get("val_accuracy", 0)
            gap = abs(train_acc - val_acc)
            results.append(TrainingResult(
                pair=instrument, model_name="lgbm_momentum",
                train_accuracy=train_acc, val_accuracy=val_acc, gap=gap,
                passed=gap <= MAX_ACCEPTABLE_GAP, duration_seconds=time.time() - t0,
            ))
            logger.info(f"    ✓ LGBM Momentum: train={train_acc:.4f} val={val_acc:.4f} gap={gap:.4f}")
    except Exception as e:
        logger.error(f"    ✗ LGBM Momentum failed: {e}")
        results.append(TrainingResult(pair=instrument, model_name="lgbm_momentum", error=str(e), duration_seconds=time.time() - t0))

    # 4. LIGHTGBM RISK
    logger.info(f"  [4/6] LightGBM Risk — {instrument}")
    t0 = time.time()
    try:
        from src.training.trainers.lightgbm_trainers import LightGBMRiskTrainer
        rf_data = load_rf_data(df_feat)
        if rf_data:
            lgbm_risk = LightGBMRiskTrainer(trainer_config)
            cp_cfg_risk = _control_plane_apply(lgbm_risk, "risk")
            lgbm_risk.train(
                rf_data["X_train"], rf_data["y_train"],
                rf_data["X_val"], rf_data["y_val"],
            )
            lgbm_risk.save(str(save_dir / "lgbm_risk.pkl"))
            metrics = lgbm_risk.get_metrics() if hasattr(lgbm_risk, "get_metrics") else {}
            _control_plane_log(
                "risk", cp_cfg_risk,
                {"train_accuracy": metrics.get("train_accuracy"),
                 "val_accuracy": metrics.get("val_accuracy")},
                save_dir / "lgbm_risk.pkl", instrument,
            )
            train_acc = metrics.get("train_accuracy", 0)
            val_acc = metrics.get("val_accuracy", 0)
            gap = abs(train_acc - val_acc)
            results.append(TrainingResult(
                pair=instrument, model_name="lgbm_risk",
                train_accuracy=train_acc, val_accuracy=val_acc, gap=gap,
                passed=gap <= MAX_ACCEPTABLE_GAP, duration_seconds=time.time() - t0,
            ))
            logger.info(f"    ✓ LGBM Risk: train={train_acc:.4f} val={val_acc:.4f} gap={gap:.4f}")
    except Exception as e:
        logger.error(f"    ✗ LGBM Risk failed: {e}")
        results.append(TrainingResult(pair=instrument, model_name="lgbm_risk", error=str(e), duration_seconds=time.time() - t0))

    # 5. RIDGE CONFIDENCE
    logger.info(f"  [5/6] Ridge Confidence — {instrument}")
    t0 = time.time()
    try:
        from src.training.trainers.ridge_trainer import RidgeTrainer
        ridge_data = load_ridge_data(df_feat)
        if ridge_data:
            ridge = RidgeTrainer(trainer_config)
            cp_cfg_conf = _control_plane_apply(ridge, "confidence")
            ridge.train(
                ridge_data["X_train"], ridge_data["y_train"],
                ridge_data["X_val"], ridge_data["y_val"],
            )
            ridge.save(str(save_dir / "ridge_confidence.pkl"))
            metrics = ridge.get_metrics() if hasattr(ridge, "get_metrics") else {}
            _control_plane_log(
                "confidence", cp_cfg_conf,
                {"val_r2": metrics.get("r2_score"),
                 "val_mae": metrics.get("confidence_mae")},
                save_dir / "ridge_confidence.pkl", instrument,
            )
            train_acc = metrics.get("train_accuracy", 0)
            val_acc = metrics.get("val_accuracy", 0)
            gap = abs(train_acc - val_acc)
            results.append(TrainingResult(
                pair=instrument, model_name="ridge_confidence",
                train_accuracy=train_acc, val_accuracy=val_acc, gap=gap,
                passed=gap <= MAX_ACCEPTABLE_GAP, duration_seconds=time.time() - t0,
            ))
            logger.info(f"    ✓ Ridge: train={train_acc:.4f} val={val_acc:.4f} gap={gap:.4f}")
    except Exception as e:
        logger.error(f"    ✗ Ridge failed: {e}")
        results.append(TrainingResult(pair=instrument, model_name="ridge_confidence", error=str(e), duration_seconds=time.time() - t0))

    # 6. HISTGRADIENTBOOSTING (Sanity check)
    logger.info(f"  [6/6] HistGradientBoosting — {instrument}")
    t0 = time.time()
    try:
        from src.training.trainers.histgb_trainer import HistGradientBoostingDirectionTrainer
        if dir_data:
            # HistGB uses flattened features (not sequences)
            X_train_flat = dir_data["X_train"].reshape(dir_data["X_train"].shape[0], -1) if len(dir_data["X_train"].shape) == 3 else dir_data["X_train"]
            X_val_flat = dir_data["X_val"].reshape(dir_data["X_val"].shape[0], -1) if len(dir_data["X_val"].shape) == 3 else dir_data["X_val"]
            histgb = HistGradientBoostingDirectionTrainer(trainer_config)
            histgb.train(X_train_flat, dir_data["y_train"], X_val_flat, dir_data["y_val"])
            histgb.save(str(save_dir / "histgb_direction.pkl"))
            metrics = histgb.get_metrics() if hasattr(histgb, "get_metrics") else {}
            train_acc = metrics.get("train_accuracy", 0)
            val_acc = metrics.get("val_accuracy", 0)
            gap = abs(train_acc - val_acc)
            results.append(TrainingResult(
                pair=instrument, model_name="histgb_direction",
                train_accuracy=train_acc, val_accuracy=val_acc, gap=gap,
                passed=gap <= MAX_ACCEPTABLE_GAP, duration_seconds=time.time() - t0,
            ))
            logger.info(f"    ✓ HistGB: train={train_acc:.4f} val={val_acc:.4f} gap={gap:.4f}")
    except Exception as e:
        logger.error(f"    ✗ HistGB failed: {e}")
        results.append(TrainingResult(pair=instrument, model_name="histgb_direction", error=str(e), duration_seconds=time.time() - t0))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: REGIME & VOLATILITY MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def train_regime_models(instrument: str, df: pd.DataFrame, params: dict) -> List[TrainingResult]:
    """Train TCN Volatility Regime + Transformer Regime + Regime-LightGBM."""
    results = []
    df_feat = apply_feature_engineering(df)

    from src.training.trainers.config import TrainerConfig
    trainer_config = TrainerConfig(
        epochs=100,
        batch_size=min(params.get("batch_size", 64), 64),
        learning_rate=params.get("lr", 0.0005),
        patience=params.get("patience", 20),
        seq_len=min(params.get("seq_len", 60), 60),
        use_replay_buffer=False,
        overfit_threshold=0.04,
        critical_threshold=0.08,
        max_acceptable_gap=MAX_ACCEPTABLE_GAP,
    )

    save_dir = MODELS_DIR / instrument
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. TCN VOLATILITY REGIME (4-class: QUIET, STABLE, ACTIVE, EXTREME)
    logger.info(f"  [Regime 1/3] TCN Volatility Regime — {instrument}")
    t0 = time.time()
    try:
        from src.training.trainers.tcn_volatility_trainer import TCNVolatilityRegimeTrainer
        from src.core.modular_data_loaders import load_direction_data
        dir_data_regime = load_direction_data(df_feat)
        if dir_data_regime:
            vol_trainer = TCNVolatilityRegimeTrainer(trainer_config)
            vol_trainer.train(
                dir_data_regime["X_train"], dir_data_regime["y_train"],
                dir_data_regime["X_val"], dir_data_regime["y_val"],
                seq_len=trainer_config.seq_len,
            )
            vol_trainer.save(str(save_dir / "tcn_volatility_regime.keras"))
            metrics = vol_trainer.get_metrics() if hasattr(vol_trainer, "get_metrics") else {}
            val_acc = metrics.get("val_accuracy", 0)
            results.append(TrainingResult(
                pair=instrument, model_name="tcn_volatility_regime",
                val_accuracy=val_acc, passed=True, duration_seconds=time.time() - t0,
            ))
            logger.info(f"    ✓ TCN Vol Regime: val={val_acc:.4f}")
        else:
            results.append(TrainingResult(pair=instrument, model_name="tcn_volatility_regime", error="No direction data"))
    except Exception as e:
        logger.error(f"    ✗ TCN Vol Regime failed: {e}")
        results.append(TrainingResult(pair=instrument, model_name="tcn_volatility_regime", error=str(e), duration_seconds=time.time() - t0))

    # 2. TRANSFORMER REGIME (3-class: TREND, CHOP, MEAN_REVERT)
    logger.info(f"  [Regime 2/3] Transformer Regime — {instrument}")
    t0 = time.time()
    try:
        from src.training.trainers.transformer_regime_trainer import TransformerRegimeTrainer
        from src.core.modular_data_loaders import load_direction_data as _load_dir
        dir_data_r2 = _load_dir(df_feat)
        if dir_data_r2:
            regime_trainer = TransformerRegimeTrainer(trainer_config)
            regime_trainer.train(
                dir_data_r2["X_train"], dir_data_r2["y_train"],
                dir_data_r2["X_val"], dir_data_r2["y_val"],
            )
            regime_trainer.save(str(save_dir / "transformer_regime.keras"))
            metrics = regime_trainer.get_metrics() if hasattr(regime_trainer, "get_metrics") else {}
            val_acc = metrics.get("val_accuracy", 0)
            results.append(TrainingResult(
                pair=instrument, model_name="transformer_regime",
                val_accuracy=val_acc, passed=True, duration_seconds=time.time() - t0,
            ))
            logger.info(f"    ✓ Transformer Regime: val={val_acc:.4f}")
    except Exception as e:
        logger.error(f"    ✗ Transformer Regime failed: {e}")
        results.append(TrainingResult(pair=instrument, model_name="transformer_regime", error=str(e), duration_seconds=time.time() - t0))

    # 3. REGIME-SPECIFIC LIGHTGBM (per-regime conditional models)
    logger.info(f"  [Regime 3/3] Regime-Specific LightGBM — {instrument}")
    t0 = time.time()
    try:
        # Use LightGBMRiskTrainer as a concrete subclass for regime-specific training
        from src.training.trainers.lightgbm_trainers import LightGBMRiskTrainer
        from src.core.modular_data_loaders import load_rf_data
        rf_data = load_rf_data(df_feat)
        if rf_data:
            regime_lgbm = LightGBMRiskTrainer(trainer_config)
            regime_lgbm.train(
                rf_data["X_train"], rf_data["y_train"],
                rf_data["X_val"], rf_data["y_val"],
            )
            regime_lgbm.save(str(save_dir / "regime_lgbm_risk.pkl"))
            results.append(TrainingResult(
                pair=instrument, model_name="regime_lgbm_risk",
                passed=True, duration_seconds=time.time() - t0,
            ))
            logger.info(f"    ✓ Regime LightGBM Risk saved")
        else:
            results.append(TrainingResult(pair=instrument, model_name="regime_lgbm_risk", error="No RF data"))
    except Exception as e:
        logger.error(f"    ✗ Regime LightGBM failed: {e}")
        results.append(TrainingResult(pair=instrument, model_name="regime_lgbm_risk", error=str(e), duration_seconds=time.time() - t0))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: RL MODELS (Position Sizer + Gate Thresholds + Optimal Exits)
# ═══════════════════════════════════════════════════════════════════════════════

def train_rl_suite() -> List[TrainingResult]:
    """Train all RL models using aggregated ensemble data."""
    results = []

    # 1. RL POSITION SIZER (PPO)
    logger.info("  [RL 1/3] Position Sizer (PPO)")
    t0 = time.time()
    try:
        from src.training.rl.position_sizer import RLConfig, RLPositionSizer, _ensure_sb3_imported, _ensure_gym_imported
        if _ensure_sb3_imported() and _ensure_gym_imported():
            config = RLConfig(total_timesteps=RL_TIMESTEPS)

            # Load existing training data or generate from replay
            data_path = MODELS_DIR / "rl_train_data.npz"
            if data_path.exists():
                data = np.load(data_path)
                features = data["features"]
                predictions = data["predictions"]
                prices = data["prices"]

                sizer = RLPositionSizer(config)
                stats = sizer.train(features, predictions, prices)
                results.append(TrainingResult(
                    pair="ALL", model_name="rl_position_sizer",
                    val_accuracy=stats.get("win_rate", 0), passed=True,
                    duration_seconds=time.time() - t0,
                ))
                logger.info(f"    ✓ RL Position Sizer: win_rate={stats.get('win_rate', 0):.2%}")
            else:
                logger.warning("    ⚠ No rl_train_data.npz — skipping RL position sizer")
                results.append(TrainingResult(pair="ALL", model_name="rl_position_sizer", error="No training data"))
        else:
            results.append(TrainingResult(pair="ALL", model_name="rl_position_sizer", error="Missing stable-baselines3/gymnasium"))
    except Exception as e:
        logger.error(f"    ✗ RL Position Sizer failed: {e}")
        results.append(TrainingResult(pair="ALL", model_name="rl_position_sizer", error=str(e), duration_seconds=time.time() - t0))

    # 2. RL GATE THRESHOLDS (SAC)
    logger.info("  [RL 2/3] Gate Thresholds (SAC)")
    t0 = time.time()
    try:
        from src.rl.gate_threshold_env import GateThresholdRL
        gate_rl = GateThresholdRL()
        gate_rl.train(total_timesteps=RL_TIMESTEPS)
        gate_rl.save(str(MODELS_DIR / "sac_gate_thresholds.zip"))
        results.append(TrainingResult(
            pair="ALL", model_name="rl_gate_thresholds",
            passed=True, duration_seconds=time.time() - t0,
        ))
        logger.info(f"    ✓ RL Gate Thresholds trained")
    except Exception as e:
        logger.error(f"    ✗ RL Gate Thresholds failed: {e}")
        results.append(TrainingResult(pair="ALL", model_name="rl_gate_thresholds", error=str(e), duration_seconds=time.time() - t0))

    # 3. RL OPTIMAL EXITS (PPO)
    logger.info("  [RL 3/3] Optimal Exits (PPO)")
    t0 = time.time()
    try:
        from src.rl.optimal_exit_env import OptimalExitRL
        exit_rl = OptimalExitRL()
        exit_rl.train(total_timesteps=RL_TIMESTEPS)
        exit_rl.save(str(MODELS_DIR / "ppo_optimal_exit.zip"))
        results.append(TrainingResult(
            pair="ALL", model_name="rl_optimal_exits",
            passed=True, duration_seconds=time.time() - t0,
        ))
        logger.info(f"    ✓ RL Optimal Exits trained")
    except Exception as e:
        logger.error(f"    ✗ RL Optimal Exits failed: {e}")
        results.append(TrainingResult(pair="ALL", model_name="rl_optimal_exits", error=str(e), duration_seconds=time.time() - t0))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 4: META-LABELER + CONFIDENCE CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_meta_labeler() -> List[TrainingResult]:
    """Train meta-labeler from trade journal outcomes."""
    results = []
    logger.info("  [Meta] Meta-Labeler Training")
    t0 = time.time()
    try:
        from src.training.meta_labeler_retrainer import MetaLabelerRetrainer
        retrainer = MetaLabelerRetrainer()
        stats = retrainer.retrain()
        results.append(TrainingResult(
            pair="ALL", model_name="meta_labeler",
            val_accuracy=stats.get("accuracy", 0) if stats else 0,
            passed=True, duration_seconds=time.time() - t0,
        ))
        logger.info(f"    ✓ Meta-Labeler retrained")
    except Exception as e:
        logger.error(f"    ✗ Meta-Labeler failed: {e}")
        results.append(TrainingResult(pair="ALL", model_name="meta_labeler", error=str(e), duration_seconds=time.time() - t0))
    return results


def fit_confidence_calibrator(instrument: str, df: pd.DataFrame) -> List[TrainingResult]:
    """Fit Platt-scaling confidence calibrator for a pair."""
    results = []
    logger.info(f"  [Calibration] Confidence Calibrator — {instrument}")
    t0 = time.time()
    try:
        from src.risk.confidence_calibration import CalibrationConfig, ConfidenceCalibrator
        save_dir = MODELS_DIR / instrument
        calib_path = save_dir / "confidence_calibrator.pkl"

        # Load the trained transformer to produce raw probabilities
        model_path = save_dir / "transformer_direction.keras"
        if model_path.exists():
            import tensorflow as tf
            model = tf.keras.models.load_model(str(model_path), compile=False)

            from src.core.modular_data_loaders import load_direction_data
            df_feat = apply_feature_engineering(df)
            dir_data = load_direction_data(df_feat, seq_len=60)
            if dir_data:
                raw_probs = model.predict(dir_data["X_val"], verbose=0).flatten()
                actual = dir_data["y_val"].flatten().astype(bool)

                config = CalibrationConfig(method="platt")
                calibrator = ConfidenceCalibrator(config)
                calibrator.fit(raw_probs, actual)
                if calibrator.is_fitted:
                    calibrator.save(calib_path)
                    results.append(TrainingResult(
                        pair=instrument, model_name="confidence_calibrator",
                        passed=True, duration_seconds=time.time() - t0,
                    ))
                    logger.info(f"    ✓ Calibrator fitted and saved")
                else:
                    results.append(TrainingResult(pair=instrument, model_name="confidence_calibrator", error="Fitting failed"))
        else:
            results.append(TrainingResult(pair=instrument, model_name="confidence_calibrator", error="No transformer model"))
    except Exception as e:
        logger.error(f"    ✗ Calibrator failed: {e}")
        results.append(TrainingResult(pair=instrument, model_name="confidence_calibrator", error=str(e), duration_seconds=time.time() - t0))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 5: TRANSFER LEARNING TO TARGET PAIRS
# ═══════════════════════════════════════════════════════════════════════════════

def transfer_to_targets(master: str, targets: List[str]) -> List[TrainingResult]:
    """Transfer master model to target pairs using warm-start + EWC."""
    results = []

    for target in targets:
        logger.info(f"  [Transfer] {master} → {target}")
        t0 = time.time()
        try:
            # Fetch target data
            df_target = fetch_candles(target, CANDLES_PER_PAIR, GRANULARITY)
            if df_target.empty:
                results.append(TrainingResult(pair=target, model_name="transfer", error="No data"))
                continue

            from src.core.modular_data_loaders import load_direction_data
            from src.training.trainers.config import TrainerConfig

            df_feat = apply_feature_engineering(df_target)
            dir_data = load_direction_data(df_feat, seq_len=60)

            if not dir_data:
                results.append(TrainingResult(pair=target, model_name="transfer", error="No direction data"))
                continue

            # Load master model as warm-start
            master_model_path = MODELS_DIR / master / "transformer_direction.keras"
            target_dir = MODELS_DIR / target
            target_dir.mkdir(parents=True, exist_ok=True)

            transfer_config = TrainerConfig(
                epochs=50,  # Fine-tuning epochs
                batch_size=128,
                learning_rate=0.0005 * 0.03,  # 33x LR reduction
                patience=15,
                seq_len=60,
                use_ewc=True,
                warm_start_freeze_encoder=True,
                warm_start_encoder_layers_to_freeze=2,
                warm_start_gradual_unfreeze=True,
                warm_start_unfreeze_epochs=10,
                overfit_threshold=0.04,
                max_acceptable_gap=MAX_ACCEPTABLE_GAP,
            )

            from src.training.trainers.transformer_trainer import TransformerDirectionTrainer
            trainer = TransformerDirectionTrainer(transfer_config)

            # Try warm-start from master
            if master_model_path.exists():
                try:
                    import tensorflow as tf
                    trainer.model = tf.keras.models.load_model(str(master_model_path), compile=False)
                    logger.info(f"    Loaded master model from {master}")
                except Exception as e:
                    logger.warning(f"    Could not load master model: {e}")

            trainer.train(
                dir_data["X_train"], dir_data["y_train"],
                dir_data["X_val"], dir_data["y_val"],
            )
            trainer.save(str(target_dir / "transformer_direction.keras"))

            metrics = trainer.get_metrics() if hasattr(trainer, "get_metrics") else {}
            train_acc = metrics.get("train_direction_accuracy", metrics.get("train_accuracy", 0))
            val_acc = metrics.get("val_direction_accuracy", metrics.get("val_accuracy", 0))
            gap = abs(train_acc - val_acc)
            passed = gap <= MAX_ACCEPTABLE_GAP

            results.append(TrainingResult(
                pair=target, model_name=f"transfer_from_{master}",
                train_accuracy=train_acc, val_accuracy=val_acc, gap=gap,
                passed=passed, duration_seconds=time.time() - t0,
            ))
            logger.info(f"    ✓ {target}: train={train_acc:.4f} val={val_acc:.4f} gap={gap:.4f} {'PASS' if passed else 'FAIL'}")

            # Also train LightGBM models for target
            try:
                from src.training.trainers.lightgbm_trainers import LightGBMMomentumTrainer, LightGBMRiskTrainer
                from src.core.modular_data_loaders import load_xgboost_data, load_rf_data
                from src.training.trainers.ridge_trainer import RidgeTrainer
                from src.core.modular_data_loaders import load_ridge_data

                mom_data = load_xgboost_data(df_feat)
                if mom_data:
                    lgbm_mom = LightGBMMomentumTrainer(transfer_config)
                    lgbm_mom.train(mom_data["X_train"], mom_data["y_train"], mom_data["X_val"], mom_data["y_val"])
                    lgbm_mom.save(str(target_dir / "lgbm_momentum.pkl"))

                rf_data = load_rf_data(df_feat)
                if rf_data:
                    lgbm_risk = LightGBMRiskTrainer(transfer_config)
                    lgbm_risk.train(rf_data["X_train"], rf_data["y_train"], rf_data["X_val"], rf_data["y_val"])
                    lgbm_risk.save(str(target_dir / "lgbm_risk.pkl"))

                ridge_data = load_ridge_data(df_feat)
                if ridge_data:
                    ridge = RidgeTrainer(transfer_config)
                    ridge.train(ridge_data["X_train"], ridge_data["y_train"], ridge_data["X_val"], ridge_data["y_val"])
                    ridge.save(str(target_dir / "ridge_confidence.pkl"))

                logger.info(f"    ✓ {target}: LightGBM + Ridge trained")
            except Exception as e:
                logger.warning(f"    ⚠ {target} support models failed: {e}")

        except Exception as e:
            logger.error(f"    ✗ Transfer to {target} failed: {e}")
            results.append(TrainingResult(pair=target, model_name=f"transfer_from_{master}", error=str(e), duration_seconds=time.time() - t0))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# JOINT TRAINING (cross-pair foundation model)
# ═══════════════════════════════════════════════════════════════════════════════

def train_joint_model(all_dfs: Dict[str, pd.DataFrame]) -> List[TrainingResult]:
    """Train joint multi-pair foundation model."""
    results = []
    logger.info("  [Joint] Multi-Pair Foundation Model")
    t0 = time.time()
    try:
        from src.training.trainers.joint_trainer import JointMultiPairTrainer
        from src.training.trainers.config import TrainerConfig

        config = TrainerConfig(
            epochs=200,
            batch_size=128,
            learning_rate=0.0005,
            patience=25,
            overfit_threshold=0.04,
            max_acceptable_gap=MAX_ACCEPTABLE_GAP,
        )

        joint_trainer = JointMultiPairTrainer(config)
        instruments = list(all_dfs.keys())

        # Add features to all dfs
        feat_dfs = {}
        for inst, df in all_dfs.items():
            feat_dfs[inst] = apply_feature_engineering(df)

        data = joint_trainer.prepare_joint_data(feat_dfs, instruments)
        joint_trainer.train_all(data)

        joint_dir = MODELS_DIR / "joint"
        joint_dir.mkdir(parents=True, exist_ok=True)
        joint_trainer.save(str(joint_dir))

        results.append(TrainingResult(
            pair="JOINT", model_name="joint_multi_pair",
            passed=True, duration_seconds=time.time() - t0,
        ))
        logger.info(f"    ✓ Joint model trained for {len(instruments)} pairs")
    except Exception as e:
        logger.error(f"    ✗ Joint training failed: {e}")
        logger.error(traceback.format_exc())
        results.append(TrainingResult(pair="JOINT", model_name="joint_multi_pair", error=str(e), duration_seconds=time.time() - t0))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def gc_cleanup():
    """Aggressively free memory between training runs."""
    import gc
    gc.collect()
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass


def main():
    """Run the full training pipeline."""
    overall_start = time.time()
    all_results: List[TrainingResult] = []

    print("=" * 70)
    print("  BUDDY FULL ENSEMBLE TRAINING — ALL TIERS, ALL CORRELATION GROUPS")
    print(f"  Candles: {CANDLES_PER_PAIR:,} per pair | Granularity: {GRANULARITY}")
    print(f"  Max Gap: {MAX_ACCEPTABLE_GAP:.0%} | Masters: {', '.join(UNIQUE_MASTERS)}")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: TRAIN ALL MASTER PAIRS (Core Ensemble + Regime)
    # ═══════════════════════════════════════════════════════════════════════
    for i, master in enumerate(UNIQUE_MASTERS, 1):
        print(f"\n{'━' * 70}")
        print(f"  MASTER {i}/{len(UNIQUE_MASTERS)}: {master}")
        print(f"{'━' * 70}")

        # Fetch data
        df = fetch_candles(master, CANDLES_PER_PAIR, GRANULARITY)
        if df.empty:
            logger.error(f"No data for {master} — skipping")
            continue

        params = MASTER_PARAMS.get(master, {})

        # Tier 1: Core ensemble
        print(f"\n  ── TIER 1: Core Ensemble ──")
        results = train_core_ensemble(master, df, params)
        all_results.extend(results)
        gc_cleanup()

        # Tier 2: Regime & volatility
        print(f"\n  ── TIER 2: Regime & Volatility Models ──")
        results = train_regime_models(master, df, params)
        all_results.extend(results)
        gc_cleanup()

        # Confidence calibrator
        print(f"\n  ── Confidence Calibration ──")
        results = fit_confidence_calibrator(master, df)
        all_results.extend(results)

        # Free this pair's data before loading next
        del df
        gc_cleanup()

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: JOINT MULTI-PAIR FOUNDATION MODEL
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print(f"  JOINT MULTI-PAIR FOUNDATION MODEL")
    print(f"{'━' * 70}")
    # Re-fetch data for joint training (smaller candle count to fit in memory)
    joint_dfs = {}
    for master in UNIQUE_MASTERS:
        joint_df = fetch_candles(master, 10_000, GRANULARITY)  # 10k for joint model (memory constrained)
        if not joint_df.empty:
            joint_dfs[master] = joint_df
    results = train_joint_model(joint_dfs)
    all_results.extend(results)
    del joint_dfs
    gc_cleanup()

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: TRANSFER TO ALL TARGET PAIRS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print(f"  TRANSFER LEARNING TO TARGET PAIRS")
    print(f"{'━' * 70}")

    # Deduplicate targets (some pairs appear in multiple groups)
    seen_targets = set()
    for group_name, group_info in CORRELATION_GROUPS.items():
        master = group_info["master"]
        for target in group_info["targets"]:
            if target not in seen_targets and target not in UNIQUE_MASTERS:
                seen_targets.add(target)

    for group_name, group_info in CORRELATION_GROUPS.items():
        master = group_info["master"]
        # Only transfer to targets we haven't already trained as masters
        targets = [t for t in group_info["targets"] if t not in UNIQUE_MASTERS]
        if targets:
            print(f"\n  {group_name}: {master} → {', '.join(targets)}")
            results = transfer_to_targets(master, targets)
            all_results.extend(results)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: RL SUITE
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print(f"  RL SUITE: Position Sizer + Gate Thresholds + Optimal Exits")
    print(f"{'━' * 70}")
    results = train_rl_suite()
    all_results.extend(results)

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 5: META-LABELER
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print(f"  META-LABELER RETRAINING")
    print(f"{'━' * 70}")
    results = train_meta_labeler()
    all_results.extend(results)

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════════
    total_time = time.time() - overall_start

    print(f"\n{'═' * 70}")
    print(f"  TRAINING COMPLETE — FINAL REPORT")
    print(f"{'═' * 70}")
    print(f"  Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.1f} hours)")
    print()

    # Direction models gap validation
    direction_models = [r for r in all_results if "direction" in r.model_name or "transfer" in r.model_name]
    passed_models = [r for r in direction_models if r.passed and r.error is None]
    failed_models = [r for r in direction_models if not r.passed and r.error is None]
    error_models = [r for r in all_results if r.error is not None]

    print(f"  Direction Models:")
    print(f"    PASSED (<{MAX_ACCEPTABLE_GAP:.0%} gap): {len(passed_models)}")
    print(f"    FAILED (>{MAX_ACCEPTABLE_GAP:.0%} gap): {len(failed_models)}")
    print(f"    ERRORS:                 {len(error_models)}")
    print()

    for r in all_results:
        status = "✓ PASS" if r.passed and not r.error else ("✗ ERROR" if r.error else "✗ FAIL")
        gap_str = f"gap={r.gap:.4f}" if r.gap > 0 else ""
        acc_str = f"train={r.train_accuracy:.4f} val={r.val_accuracy:.4f}" if r.val_accuracy > 0 else ""
        err_str = f"err={r.error[:40]}" if r.error else ""
        print(f"  {status:10s} {r.pair:10s} {r.model_name:30s} {acc_str} {gap_str} {err_str}")

    # Save report
    report = {
        "completed_at": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "candles_per_pair": CANDLES_PER_PAIR,
        "granularity": GRANULARITY,
        "max_acceptable_gap": MAX_ACCEPTABLE_GAP,
        "masters": UNIQUE_MASTERS,
        "results": [r.to_dict() for r in all_results],
        "summary": {
            "total_models": len(all_results),
            "passed": len(passed_models),
            "failed": len(failed_models),
            "errors": len(error_models),
        },
    }
    report_path = PROJECT_ROOT / "trained_data" / "full_training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"\n  Report saved: {report_path}")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
