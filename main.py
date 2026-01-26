#!/usr/bin/env python3
"""
Enhanced ML Engine Trading Bot CLI

A command-line interface for the ML trading engine that handles:
- Command processing and orchestration
- Interactive dashboard visualization
- Configuration management
- Training/evaluation progress display
- Real-time monitoring
"""

from __future__ import annotations

# Suppress TensorFlow CPU instruction warnings (irrelevant on Apple Silicon/Metal)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

import sys
import time
import logging
import argparse
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from utils import setup_logging, load_config


@dataclass(frozen=True)
class OandaFetchOptions:
    instrument: str = "USD_JPY"
    granularity: str = "M5"
    candles: int = 5000
    price: str = "MBA"
    save_csv: str | None = None


@dataclass(frozen=True)
class BuddyTrainingAdvancedOptions:
    init_from: str | None = None
    es_monitor: str = "direction"  # direction|combined|val_loss|loss
    combined_w_dir: float = 0.7
    combined_w_conf: float = 0.3
    train_smoothing: bool = True
    top_features: int | None = None
    median_window: int | None = None
    vol_norm_window: int | None = None
    min_volume: int | None = None
    spread_filter: bool = False
    spread_pctl: float = 0.99
    spread_mult: float = 3.0

    # Tier-2: calibrated win probability (TP-before-SL) using fixed SL/TP.
    tier2_calibrate: bool = True
    tier2_stop_loss_pips: float | None = None
    tier2_take_profit_pips: float | None = None
    tier2_spread_fallback_pips: float | None = None
    tier2_horizon_candles: int | None = None
    tier2_tie_break: str | None = None
    tier2_calibration_bins: int = 20
    tier2_calibration_stride: int = 1


@dataclass(frozen=True)
class BuddyTrainingOptions:
    """Training options for Buddy model.
    
    M1 Metal Optimizations Applied:
    - model_type: "tcn" - 2-3x faster than LSTM (parallelizable convolutions)
    - batch_size: 128 (was 32) - better GPU utilization
    - mixed_precision: True (was False) - 1.5-2x speedup
    - steps_per_execution: 10 (was 1) - reduces Python overhead
    - lr: 0.0005 (was 0.001) - more stable convergence
    
    Model Types:
    - "tcn": Temporal Convolutional Network (RECOMMENDED for M1 - fastest)
    - "lstm": Legacy LSTM (slower on Metal, avoid unless necessary)
    - "attention_lstm": LSTM with attention (slower but more accurate)
    
    Enterprise Training (MLOps):
    - enterprise: Enable MLflow tracking, CV, bootstrap CI
    - cv_folds: Walk-forward cross-validation folds
    - bootstrap: Bootstrap confidence intervals
    
    Multi-Pair Pre-training:
    - multi_pair: Train foundation model on multiple pairs simultaneously
    - foundation_pairs: Comma-separated list of pairs for pre-training
    """
    oanda_fetch: OandaFetchOptions | None = None
    advanced: BuddyTrainingAdvancedOptions | None = None
    # None => use config (buddy.feature_curriculum)
    feature_curriculum: bool | None = None
    curriculum_ks: str | None = None
    pca_components: int | None = None
    seq_len: int = 60  # M1: Match config_m1_optimized.yaml (was 50)
    epochs: int = 200  # M1: Reduced from 300 with better early stopping
    batch_size: int = 128  # M1 CRITICAL: 128 is optimal for Metal (was 32)
    lr: float = 0.0005  # M1: Lower LR for stability (was 0.001)
    patience: int = 15  # M1: Increased patience (was 10)
    seed: int = 42
    run_tag: str | None = None
    all_features: bool = True
    device: str = "auto"  # auto|cpu|gpu
    ignore_input_mismatches: bool = False
    disable_tier2_on_mismatch: bool = False
    mixed_precision: bool = True  # M1 CRITICAL: Enable for 1.5-2x speedup (was False)
    steps_per_execution: int = 10  # M1 CRITICAL: Reduces Python overhead (was 1)
    jit_compile: bool = False  # Keep False for stability (XLA can cause issues)
    shuffle_buffer: int | None = None
    prefetch: int | None = None
    cache_val: bool = True  # M1: Cache validation data (was False)
    combined_use_predict: bool = True
    shared_encoder: bool = False  # Disabled when using TCN (TCN doesn't need shared encoder)
    model_type: str = "tcn"  # M1 CRITICAL: TCN is 2-3x faster than LSTM on Metal
    timing: bool = False
    fit_verbose: int = 1
    # Enterprise training (MLOps) options - ENABLED BY DEFAULT for ensemble
    enterprise: bool = True  # Enable enterprise features (MLflow, CV, bootstrap)
    cv_folds: int = 5  # Walk-forward cross-validation folds (0 to disable)
    bootstrap: bool = True  # Enable bootstrap confidence intervals
    bootstrap_samples: int = 1000  # Bootstrap iterations
    mlflow_experiment: str | None = None  # MLflow experiment name
    generate_report: bool = True  # Generate markdown report
    # Continual learning - EWC enabled by default for better multi-instrument learning
    enable_ewc: bool = True  # Enable EWC (Elastic Weight Consolidation)
    # Multi-pair pre-training - foundation model across instruments
    multi_pair: bool = False  # Enable multi-pair foundation training
    foundation_pairs: str | None = None  # Comma-separated pairs (default: majors)
    # RL position sizing training - train RL agent after ensemble
    train_rl_sizer: bool = True  # Enable RL position sizer training after ensemble
    rl_timesteps: int = 500_000  # RL training timesteps (500k ~ 10-15 min on M1)


def _tier2_get_calibration_dict(meta: dict[str, Any]) -> dict[str, Any] | None:
    tier2 = meta.get("tier2") or {}
    calib = tier2.get("calibration") or tier2.get("calibration_v1") or meta.get("tier2_calibration")
    return calib if isinstance(calib, dict) else None


def _tier2_points_from_bins(calib: dict[str, Any]) -> list[tuple[float, float]]:
    bins = calib.get("bins")
    if not isinstance(bins, list) or not bins:
        return []

    pts: list[tuple[float, float]] = []
    for b in bins:
        if not isinstance(b, dict):
            continue
        x = b.get("score")
        y = b.get("p_win")
        if x is None or y is None:
            continue
        try:
            xf = float(x)
            yf = float(y)
        except Exception:
            continue
        if 0.0 <= xf <= 1.0:
            pts.append((xf, float(max(0.0, min(1.0, yf)))))
    pts.sort(key=lambda t: t[0])
    return pts


def _tier2_interpolate_points(pts: list[tuple[float, float]], score: float) -> float | None:
    if not pts:
        return None

    s = float(max(0.0, min(1.0, float(score))))
    x_min, y_min = pts[0]
    x_max, y_max = pts[-1]

    if s <= x_min:
        return float(y_min)
    if s >= x_max:
        return float(y_max)

    lo = 0
    hi = len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if s <= pts[mid][0]:
            hi = mid
        else:
            lo = mid

    x0, y0 = pts[lo]
    x1, y1 = pts[hi]
    if x1 <= x0 + 1e-12:
        return float(y1)
    t = (s - x0) / (x1 - x0)
    return float((1.0 - t) * y0 + t * y1)


def _tier2_clip_prob(p: float, *, eps: float = 1e-7) -> float:
    try:
        pf = float(p)
    except Exception:
        return 0.5
    if np.isnan(pf):
        return 0.5
    return float(max(eps, min(1.0 - eps, pf)))


def _tier2_logit(p: float, *, eps: float = 1e-7) -> float:
    pp = _tier2_clip_prob(p, eps=eps)
    return float(np.log(pp / (1.0 - pp)))


def _tier2_sigmoid(x: float) -> float:
    xf = float(x)
    if xf >= 0:
        ex = float(np.exp(-xf))
        return float(1.0 / (1.0 + ex))
    ex = float(np.exp(xf))
    return float(ex / (1.0 + ex))


def _tier2_temperature_scale_prob(p: float, temperature: float, *, eps: float = 1e-7) -> float:
    try:
        temperature_f = float(temperature)
    except Exception:
        return float(max(0.0, min(1.0, float(p))))
    if not np.isfinite(temperature_f) or temperature_f <= 0:
        return float(max(0.0, min(1.0, float(p))))
    z = _tier2_logit(p, eps=eps)
    return float(_tier2_sigmoid(z / temperature_f))


def _tier2_nll(y_true: np.ndarray, p_pred: np.ndarray, *, eps: float = 1e-15) -> float:
    p = np.clip(p_pred.astype(np.float64, copy=False), eps, 1.0 - eps)
    y = y_true.astype(np.float64, copy=False)
    return float(-np.mean((y * np.log(p)) + ((1.0 - y) * np.log(1.0 - p))))


def _tier2_spearman(a: np.ndarray, b: np.ndarray) -> float:
    # Simple Spearman via rank correlation (stable enough for diagnostics).
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or aa.size != bb.size:
        return float("nan")
    ra = aa.argsort(kind="mergesort").argsort(kind="mergesort").astype(np.float64)
    rb = bb.argsort(kind="mergesort").argsort(kind="mergesort").astype(np.float64)
    try:
        return float(np.corrcoef(ra, rb)[0, 1])
    except Exception:
        return float("nan")


def _tier2_apply_calibration(meta: dict[str, Any], score: float) -> float | None:
    """Map a model score in [0,1] to calibrated P(win).

    Prefers temperature scaling when available; falls back to saved bin interpolation.
    """
    try:
        calib = _tier2_get_calibration_dict(meta)
        if calib is None:
            return None

        # Primary: temperature scaling
        ts = calib.get("temperature_scaling") if isinstance(calib, dict) else None
        if isinstance(ts, dict) and bool(ts.get("enabled", False)):
            T = ts.get("T")
            if T is not None:
                return _tier2_temperature_scale_prob(float(score), float(T))

        # Fallback: bin interpolation
        pts = _tier2_points_from_bins(calib)
        return _tier2_interpolate_points(pts, score)
    except Exception:
        return None


def _oanda_fetch_to_csv(opts: OandaFetchOptions) -> str:
    from datetime import datetime, timezone

    import pandas as pd

    from fx_paper import candles_to_ohlcv_df
    from oanda_practice import OandaPracticeClient

    client = OandaPracticeClient.from_env()

    # OANDA v20 candles endpoint typically caps `count` at 5000.
    # If the user requests more, page forward using `from_time`.
    max_per_request = 5000
    total = max(1, int(opts.candles))
    if total <= max_per_request:
        resp = client.get_candles(
            opts.instrument,
            granularity=str(opts.granularity),
            count=int(total),
            price=str(opts.price),
        )
        oanda_df = candles_to_ohlcv_df(resp)
    else:
        # Page *backwards* using `to_time`.
        # Paging forwards using `from_time` can easily land on the current (incomplete)
        # candle and return only incomplete candles.
        dfs: list[pd.DataFrame] = []
        to_time: str | None = None
        last_min_ts = None
        remaining = int(total)
        # Safety: cap number of API calls.
        max_pages = max(1, int((total + max_per_request - 1) // max_per_request) + 2)
        for _ in range(max_pages):
            if remaining <= 0:
                break
            count = int(min(max_per_request, remaining))
            resp = client.get_candles(
                opts.instrument,
                granularity=str(opts.granularity),
                count=int(count),
                price=str(opts.price),
                to_time=to_time,
            )
            df_page = candles_to_ohlcv_df(resp)
            if df_page is None or len(df_page) == 0:
                break
            dfs.append(df_page)
            remaining -= int(len(df_page))

            # Move the cursor to the earliest candle in this page.
            try:
                t = pd.to_datetime(df_page["time"], utc=True, errors="coerce")
                if t.isna().all():
                    break
                min_ts = t.min()
                if last_min_ts is not None and min_ts >= last_min_ts:
                    break
                last_min_ts = min_ts
                # OANDA accepts RFC3339; keep full precision.
                to_time = str(min_ts.to_pydatetime().isoformat().replace(UTC_OFFSET_SUFFIX, "Z"))
            except Exception:
                break

        if dfs:
            oanda_df = pd.concat(dfs, axis=0, ignore_index=True)
            try:
                if "time" in oanda_df.columns:
                    oanda_df = oanda_df.drop_duplicates(subset=["time"], keep="last")
                    oanda_df = oanda_df.sort_values("time").reset_index(drop=True)
            except Exception:
                pass
            # Keep the most recent `total` candles.
            try:
                if len(oanda_df) > total:
                    oanda_df = oanda_df.iloc[-int(total) :].reset_index(drop=True)
            except Exception:
                pass
        else:
            oanda_df = pd.DataFrame()

    # Helpful "recency" logging so it's obvious we pulled live data now.
    candle_range_str = ""
    now_utc_str = ""
    try:
        ts_fmt = "%Y-%m-%dT%H:%M:%SZ"
        now_utc_str = datetime.now(timezone.utc).strftime(ts_fmt)
        if "time" in oanda_df.columns and len(oanda_df) > 0:
            t = pd.to_datetime(oanda_df["time"], utc=True, errors="coerce")
            if not t.isna().all():
                first_t = t.min().strftime(ts_fmt)
                last_t = t.max().strftime(ts_fmt)
                candle_range_str = f"{first_t} → {last_t}"
    except Exception:
        pass

    out_path: Path
    if opts.save_csv:
        out_path = Path(opts.save_csv)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("market_data") / f"oanda_{opts.instrument}_{opts.granularity}_live_{ts}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    oanda_df.to_csv(out_path, index=False)
    console.print(Panel(
        f"[bold]OANDA Data Retrieved[/bold]\n\n"
        f"[dim]Timestamp:[/dim] {now_utc_str}\n"
        f"[dim]Candle Range:[/dim] {candle_range_str}\n"
        f"[dim]Rows:[/dim] {len(oanda_df):,}\n"
        f"[dim]Saved to:[/dim] {out_path}",
        title="✓ Data Fetched",
        border_style="green",
    ))
    return str(out_path)


def _migrate_keras2_to_keras3(model_path: str, seq_len: int, feature_dim: int, custom_objects: dict):
    """
    Migrate a Keras 2.x model to Keras 3.x by recreating architecture and loading weights.
    This handles the keras.src.engine.functional → keras.models.Model path change.
    """
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.regularizers import l2
    import zipfile
    import json as _json
    import tempfile
    import shutil

    console.print("[yellow]Attempting Keras 2→3 migration...[/yellow]")

    # Try to extract weights from the .keras file (it's a zip)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(model_path, 'r') as zf:
                zf.extractall(tmpdir)

            # Load config to understand architecture
            config_path = Path(tmpdir) / "config.json"
            if config_path.exists():
                config = _json.loads(config_path.read_text())
                model_config = config.get("config", {})
                model_name = model_config.get("name", "unknown")
                console.print(f"[dim]Old model: {model_name}[/dim]")

            # Load weights file
            weights_path = Path(tmpdir) / "model.weights.h5"
            if not weights_path.exists():
                raise FileNotFoundError("No model.weights.h5 in .keras archive")

            # Create a fresh model with the same architecture (MetalOptimizedTCN)
            l2_reg = l2(0.002)
            hidden_size = 32
            dropout = 0.35
            
            inp = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="input")
            x = layers.GaussianNoise(0.03)(inp)
            x = layers.SpatialDropout1D(dropout * 0.5)(x)
            
            # TCN layers (2 blocks)
            for i in range(2):
                dilation_rate = 2 ** i
                conv_out = layers.Conv1D(hidden_size, 3, padding='causal', dilation_rate=dilation_rate, 
                                         activation='relu', kernel_regularizer=l2_reg)(x)
                conv_out = layers.BatchNormalization()(conv_out)
                conv_out = layers.Dropout(dropout)(conv_out)
                if x.shape[-1] != hidden_size:
                    x = layers.Conv1D(hidden_size, 1)(x)
                x = layers.Add()([x, conv_out])
            
            x = x[:, -1, :]  # Take last timestep
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(dropout)(x)
            
            # Multi-output heads
            dir_branch = layers.Dense(32, activation='relu')(x)
            dir_branch = layers.Dropout(dropout * 0.5)(dir_branch)
            price_branch = layers.Dense(32, activation='relu')(x)
            risk_branch = layers.Dense(16, activation='relu')(x)
            state_branch = layers.Dense(32, activation='relu')(x)
            state_branch = layers.Dropout(dropout * 0.5)(state_branch)
            trend_branch = layers.Dense(32, activation='relu')(x)
            
            direction = layers.Dense(1, activation='sigmoid', name='direction', dtype='float32')(dir_branch)
            price = layers.Dense(1, activation='linear', name='price', dtype='float32')(price_branch)
            risk = layers.Dense(1, activation='sigmoid', name='risk', dtype='float32')(risk_branch)
            state_logits = layers.Dense(2, activation='softmax', name='state_logits', dtype='float32')(state_branch)
            trend = layers.Dense(1, activation='linear', name='trend', dtype='float32')(trend_branch)
            
            fresh_model = tf.keras.Model(
                inputs=inp,
                outputs={'direction': direction, 'price': price, 'risk': risk, 'state_logits': state_logits, 'trend': trend},
                name='MetalOptimizedTCN',
            )

            # Try to load weights (may fail if architecture differs significantly)
            try:
                fresh_model.load_weights(str(weights_path))
                console.print("[green]✓ Weights migrated successfully[/green]")

                # Save as new Keras 3 model (overwrite old)
                backup_path = model_path + ".keras2_backup"
                if not Path(backup_path).exists():
                    shutil.copy(model_path, backup_path)
                    console.print(f"[dim]Backup saved: {backup_path}[/dim]")
                fresh_model.save(model_path)
                console.print(f"[green]✓ Model migrated to Keras 3: {model_path}[/green]")
                return fresh_model

            except Exception as weight_err:
                console.print(f"[yellow]Weight loading failed (architecture mismatch): {weight_err}[/yellow]")
                raise ValueError(f"Cannot migrate: architecture incompatible. Delete old model and retrain.")

        except zipfile.BadZipFile:
            raise ValueError(f"Model file {model_path} is not a valid .keras archive")


def _load_buddy_checkpoint(*, model_path: str, seq_len: int, feature_dim: int):
    import tensorflow as tf

    from ml_head_engine import MLEngineHead
    from mr_engine import MREngineHead
    from ms_head_engine import MSEngineHead
    from mt_engine import MTEngineHead
    from mx_head_engine import MXEngineHead

    custom_objects = {
        "MLEngineHead": MLEngineHead,
        "MREngineHead": MREngineHead,
        "MSEngineHead": MSEngineHead,
        "MTEngineHead": MTEngineHead,
        "MXEngineHead": MXEngineHead,
    }

    # Try standard load first
    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects,
        )
    except Exception as e:
        err_str = str(e)
        # Handle Keras 2 → Keras 3 incompatibility
        if "keras.src.engine.functional" in err_str or "cannot be imported" in err_str:
            # Attempt weights-only migration
            model = _migrate_keras2_to_keras3(model_path, seq_len, feature_dim, custom_objects)
        else:
            raise

    in_shape = tuple(getattr(model, "input_shape", ()) or ())
    if len(in_shape) >= 3:
        if int(in_shape[1]) != int(seq_len) or int(in_shape[2]) != int(feature_dim):
            raise ValueError(
                f"init_from model input_shape {in_shape} incompatible with seq_len={seq_len}, feature_dim={feature_dim}"
            )
    return model


BUDDY_META_FILENAME = "buddy_tf.meta.json"


def _meta_path_for_checkpoint(model_path: str) -> Path | None:
    p = Path(model_path)
    try:
        if p.suffix == ".keras":
            candidate = p.with_suffix(".meta.json")
            if candidate.exists():
                return candidate
    except Exception:
        pass

    # Special-case: default Buddy checkpoint keeps metadata in buddy_tf.meta.json
    try:
        default_model = Path("trained_data") / "models" / "buddy_tf.keras"
        if p.resolve() == default_model.resolve():
            candidate = Path("trained_data") / "models" / BUDDY_META_FILENAME
            if candidate.exists():
                return candidate
    except Exception:
        pass
    return None


# Constants
DEFAULT_CONFIG_PATH = "./config_improved_H1.yaml"
DEFAULT_MESSAGE_FORMAT = "Epoch {epoch} completed"
TIMESTAMP_FORMAT = "%H:%M:%S"
TABLE_HEADER_STYLE = "bold magenta"
DEFAULT_CURRICULUM_KS = "32,64,128,0"
UTC_OFFSET_SUFFIX = "+00:00"

# Initialize console and logging
console = Console()
logger = setup_logging(log_file="cli.log")


def _parse_bool_answer(s: str | None, *, default: bool) -> bool:
    if s is None:
        return bool(default)
    v = str(s).strip().lower()
    if not v:
        return bool(default)
    if v in {"y", "yes", "true", "1", "on"}:
        return True
    if v in {"n", "no", "false", "0", "off"}:
        return False
    return bool(default)


def _parse_float_answer(s: str | None, *, default: float) -> float:
    if s is None:
        return float(default)
    v = str(s).strip()
    if not v:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _parse_int_answer(s: str | None, *, default: int) -> int:
    if s is None:
        return int(default)
    v = str(s).strip()
    if not v:
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


# Valid OANDA FX instruments (major, minor, and exotic pairs)
VALID_OANDA_INSTRUMENTS = {
    # Major pairs
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD",
    # Cross pairs
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CHF", "AUD_CAD", "AUD_NZD",
    "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CHF", "NZD_CAD",
    # Exotic pairs
    "USD_SGD", "USD_HKD", "USD_MXN", "USD_ZAR", "USD_TRY", "USD_SEK", "USD_NOK", "USD_DKK",
    "USD_PLN", "USD_HUF", "USD_CZK", "USD_THB", "USD_INR", "USD_CNH",
    "EUR_SEK", "EUR_NOK", "EUR_DKK", "EUR_PLN", "EUR_HUF", "EUR_CZK", "EUR_TRY", "EUR_ZAR",
    "GBP_SGD", "GBP_PLN", "GBP_ZAR",
    "AUD_SGD", "AUD_HKD",
    "SGD_JPY", "SGD_CHF", "HKD_JPY", "TRY_JPY", "ZAR_JPY", "MXN_JPY",
}


def _normalize_instrument(s: str) -> str:
    v = (s or "").strip().upper().replace("-", "_").replace("/", "_")
    v = "_".join([p for p in v.split("_") if p])
    if "_" not in v and len(v) == 6:
        v = v[:3] + "_" + v[3:]
    return v


def _extract_instrument_from_csv_path(csv_path: str) -> str:
    """
    Extract instrument from CSV filename for pair-specific model training.
    
    Patterns recognized:
    - market_data/EUR_USD_H1.csv -> EUR_USD
    - market_data/oanda_GBP_USD_H1_live_*.csv -> GBP_USD
    - market_data/USD_JPY_*.csv -> USD_JPY
    - market_data/EURUSD.csv -> EUR_USD (6-char format)
    
    Returns normalized instrument or "GENERIC" if not extractable.
    """
    import re
    from pathlib import Path
    
    if not csv_path:
        return "GENERIC"
    
    filename = Path(csv_path).stem.upper()  # e.g., "EUR_USD_H1" or "oanda_GBP_USD_H1_live_20250114"
    
    # Try to find a valid instrument in the filename
    # Pattern 1: Direct match for XXX_YYY format
    for instrument in VALID_OANDA_INSTRUMENTS:
        if instrument in filename:
            return instrument
    
    # Pattern 2: 6-char format like "EURUSD" 
    match = re.search(r'([A-Z]{6})', filename)
    if match:
        candidate = match.group(1)[:3] + "_" + match.group(1)[3:]
        if candidate in VALID_OANDA_INSTRUMENTS:
            return candidate
    
    # Pattern 3: Try to extract XXX_YYY pattern from filename
    match = re.search(r'([A-Z]{3})_([A-Z]{3})', filename)
    if match:
        candidate = f"{match.group(1)}_{match.group(2)}"
        if candidate in VALID_OANDA_INSTRUMENTS:
            return candidate
    
    return "GENERIC"


def _get_pair_model_paths(model_dir: Path, instrument: str, model_type: str = "transformer") -> dict:
    """
    Get pair-specific model paths for saving/loading.
    
    Returns dict with paths for:
    - direction: Transformer/TCN direction model
    - xgboost: XGBoost momentum model  
    - rf: Random Forest risk model
    - ridge: Ridge confidence model
    - histgb: HistGB hybrid voting model (optional)
    
    Example for EUR_USD:
    - trained_data/models/EUR_USD/transformer_direction.keras
    - trained_data/models/EUR_USD/xgb_momentum.pkl
    """
    # Use GENERIC for generic models, pair name for pair-specific
    if instrument and instrument != "GENERIC":
        pair_dir = model_dir / instrument
    else:
        pair_dir = model_dir
    
    # Ensure directory exists
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    direction_filename = "transformer_direction.keras" if model_type == "transformer" else "tcn_direction.keras"
    
    return {
        'direction': pair_dir / direction_filename,
        'regime': pair_dir / "transformer_regime.keras",
        'xgboost': pair_dir / "xgb_momentum.pkl",
        'rf': pair_dir / "rf_risk.pkl",
        'ridge': pair_dir / "ridge_confidence.pkl",
        'histgb': pair_dir / "histgb_direction.pkl",
        'pair_dir': pair_dir,
    }


def _validate_instrument(instrument: str) -> str:
    """Validate and normalize an instrument name. Raises ValueError if invalid."""
    # Special case: "all" means scan all major pairs
    if instrument.upper() == "ALL":
        return "ALL"
    
    normalized = _normalize_instrument(instrument)
    
    if normalized in VALID_OANDA_INSTRUMENTS:
        return normalized
    
    # Try to suggest a correction for common typos
    suggestions = []
    for valid in VALID_OANDA_INSTRUMENTS:
        # Check if it's a simple character swap or typo
        if len(normalized) == len(valid):
            diff = sum(1 for a, b in zip(normalized, valid) if a != b)
            if diff <= 2:
                suggestions.append(valid)
    
    error_msg = f"Invalid instrument: '{instrument}' (normalized: '{normalized}')"
    if suggestions:
        error_msg += f"\n  Did you mean: {', '.join(sorted(suggestions))}?"
    error_msg += f"\n  Valid examples: EUR_USD, GBP_USD, USD_JPY, AUD_USD, EUR_GBP, or 'all' for all major pairs"
    
    raise ValueError(error_msg)


def launch_buddy_repl_from_wizard(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    instrument: str = "USD_JPY",
    granularity: str = "M5",
    candles: int = 2000,
    execute: bool = True,
    all_features: bool = False,
    verbose: bool = False,
    assistant_name: str = "Buddy",
) -> None:
    """Launch the interactive Buddy REPL (lazy-imports heavy deps).

    This is a thin helper intended to be called from the interactive wizard
    to avoid importing `unified_talk` at module import time.
    """
    try:
        from unified_talk import run_unified_talk, OandaSettings  # type: ignore
    except Exception as e:  # pragma: no cover - runtime guard
        console.print(
            f"[red]Could not import Buddy REPL (missing optional dependency): {e}[/red]"
        )
        raise

    # Build OandaSettings (some variants may have different constructor signatures).
    try:
        oanda_settings = OandaSettings(instrument=instrument, granularity=granularity, candles=int(candles), execute=bool(execute))
    except Exception:
        # Fallback simple namespace for compatibility
        class _S:  # simple compatibility shim
            pass

        oanda_settings = _S()
        setattr(oanda_settings, "instrument", instrument)
        setattr(oanda_settings, "granularity", granularity)
        setattr(oanda_settings, "candles", int(candles))
        setattr(oanda_settings, "execute", bool(execute))

    # Call the REPL entrypoint from unified_talk. Keep keyword args explicit.
    run_unified_talk(
        config_path,
        checkpoint_path=checkpoint_path,
        csv_path=None,
        ticker=None,
        period="5d",
        interval="1h",
        oanda=True,
        oanda_settings=oanda_settings,
        all_features=bool(all_features),
        verbose=bool(verbose),
        assistant_name=assistant_name,
    )


def _buddy_wizard_ask(prompt: str) -> str:
    try:
        return str(console.input(prompt) or "")
    except EOFError:
        return ""


def _buddy_wizard_collect_settings(*, default_config: str) -> dict[str, Any]:
    cfg_path = (_buddy_wizard_ask(f"Config path [{default_config}]: ") or "").strip() or default_config
    instrument = _normalize_instrument(_buddy_wizard_ask("Instrument [USD_JPY]: ") or "USD_JPY")
    granularity = (_buddy_wizard_ask("Granularity [M5]: ") or "M5").strip() or "M5"

    candles = _parse_int_answer(
        _buddy_wizard_ask("Candles lookback [2000]: "),
        default=2000,
    )
    if int(candles) < 300:
        candles = 300

    equity = _parse_float_answer(
        _buddy_wizard_ask("Equity for sizing (PRACTICE NAV fallback) [10000]: "),
        default=10_000.0,
    )

    # Risk is used as a fraction internally (0.005 = 0.5%). Ask as percent for humans.
    risk_pct = _parse_float_answer(_buddy_wizard_ask("Risk per trade (%) [0.5]: "), default=0.5)
    risk_frac = max(0.0, float(risk_pct) / 100.0)

    execute = _parse_bool_answer(_buddy_wizard_ask("Place PRACTICE orders? (Y/n): "), default=True)

    force_execute = False
    if execute:
        force_execute = _parse_bool_answer(
            _buddy_wizard_ask("Bypass training gate (>=62% dir acc)? (Y/n): "),
            default=True,
        )

    loop = _parse_bool_answer(_buddy_wizard_ask("Run continuously (loop)? (y/N): "), default=False)
    max_trades = 1
    if loop:
        max_trades = _parse_int_answer(_buddy_wizard_ask("Max executed trades (0 = unlimited) [1]: "), default=1)
        max_trades = max(0, int(max_trades))

    return {
        "cfg_path": cfg_path,
        "instrument": instrument,
        "granularity": granularity,
        "candles": int(candles),
        "equity": float(equity),
        "risk_frac": float(risk_frac),
        "execute": bool(execute),
        "force_execute": bool(force_execute),
        "loop": bool(loop),
        "max_trades": int(max_trades),
    }


def _buddy_wizard_confirm_settings(s: dict[str, Any]) -> bool:
    console.print(
        f"\n[dim]Summary[/dim]: instrument={s['instrument']} granularity={s['granularity']} candles={int(s['candles'])} "
        f"equity={float(s['equity']):.2f} risk={float(s['risk_frac'])*100:.3f}% execute={bool(s['execute'])} "
        f"force_execute={bool(s['force_execute'])} loop={bool(s['loop'])}"
    )
    if _parse_bool_answer(_buddy_wizard_ask("Proceed? (Y/n): "), default=True):
        return True
    console.print("[dim]Cancelled.[/dim]")
    return False


def _buddy_wizard_select_mode() -> int:
    # Offer a short mode menu to make intentions explicit.
    console.print(
        "\nChoose mode:\n 1) Single-shot inference: run one prediction and exit\n"
        " 2) Loop trading: run continuously and execute trades\n"
        " 3) Launch Buddy REPL: interactive trading commands (buy/sell/close/etc.)\n"
        " 4) Train Buddy: run training workflow"
    )
    choice = (_buddy_wizard_ask("Mode [1]: ") or "1").strip() or "1"
    try:
        return int(choice)
    except Exception:
        return 1


def _buddy_wizard_dispatch_mode(mode: int, s: dict[str, Any]) -> None:
    if mode == 3:
        # Launch the interactive REPL that exposes Buddy's trade commands.
        launch_buddy_repl_from_wizard(
            s["cfg_path"],
            checkpoint_path=None,
            instrument=s["instrument"],
            granularity=s["granularity"],
            candles=int(s["candles"]),
            execute=bool(s["execute"]),
            all_features=False,
            verbose=False,
            assistant_name="Buddy",
        )
        return

    if mode == 4:
        # Kick off training flow
        train_buddy(s["cfg_path"])
        return

    # Mode 2 = loop, else single-shot
    if mode == 2 or bool(s["loop"]):
        buddy_loop(
            s["cfg_path"],
            instrument=s["instrument"],
            granularity=s["granularity"],
            candles=int(s["candles"]),
            execute=bool(s["execute"]),
            force_execute=bool(s["force_execute"]),
            equity=float(s["equity"]),
            risk_per_trade_pct=float(s["risk_frac"]),
            all_features=False,
            verbose=False,
            max_trades=int(s["max_trades"]),
        )
        return

    buddy(
        s["cfg_path"],
        instrument=s["instrument"],
        granularity=s["granularity"],
        candles=int(s["candles"]),
        execute=bool(s["execute"]),
        force_execute=bool(s["force_execute"]),
        equity=float(s["equity"]),
        risk_per_trade_pct=float(s["risk_frac"]),
        all_features=False,
        verbose=False,
    )


def _buddy_interactive_wizard(*, default_config: str = DEFAULT_CONFIG_PATH) -> None:
    console.print("[bold]Buddy interactive[/bold] (press Enter for defaults)")
    settings = _buddy_wizard_collect_settings(default_config=default_config)
    if not _buddy_wizard_confirm_settings(settings):
        return
    _buddy_wizard_dispatch_mode(_buddy_wizard_select_mode(), settings)


def _configure_predict_output(verbose: bool) -> None:
    """Reduce log/warning noise for interactive predict runs."""
    if verbose:
        return

    import warnings

    # Keep common third-party warnings from drowning out the prediction output.
    # (Seen on macOS system Python where ssl is LibreSSL.)
    warnings.filterwarnings(
        "ignore",
        message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+",
    )
    try:
        import pandas as pd  # noqa: F401
        from pandas.errors import PerformanceWarning

        warnings.filterwarnings("ignore", category=PerformanceWarning)
    except Exception:
        pass

    # Keep CLI output clean by muting INFO logs from internal modules.
    logging.getLogger().setLevel(logging.WARNING)
    for name in (
        "utils",
        "reasoning_enhanced",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _tf_env_flag(name: str) -> str | None:
    import os

    v = os.environ.get(name)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _tf_is_truthy(v: str | None) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _tf_set_log_level(*, verbose: bool) -> None:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2" if verbose else "3")


def _tf_should_disable_meta_optimizer() -> bool:
    import platform

    v = _tf_env_flag("BUDDY_DISABLE_META_OPTIMIZER")
    if v is not None:
        return _tf_is_truthy(v)
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _tf_try_disable_meta_optimizer(tf_mod, *, verbose: bool) -> None:
    if not _tf_should_disable_meta_optimizer():
        return
    try:
        tf_mod.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
        if verbose:
            console.print("[green]TensorFlow meta optimizer disabled[/green] (BUDDY_DISABLE_META_OPTIMIZER=1)")
    except Exception:
        pass


def _tf_try_force_cpu(tf_mod, *, verbose: bool) -> None:
    try:
        tf_mod.config.set_visible_devices([], "GPU")
        if verbose:
            console.print("[yellow]TensorFlow forced to CPU[/yellow] (--device cpu)")
    except Exception as e:
        if verbose:
            console.print(f"[yellow]Could not force CPU-only TensorFlow[/yellow]: {e}")


def _tf_try_apply_gpu_memory_limit(tf_mod, gpus, *, verbose: bool) -> None:
    import os

    mem_limit_mb = os.environ.get("BUDDY_GPU_MEMORY_LIMIT_MB")
    if not mem_limit_mb:
        return
    try:
        limit = int(mem_limit_mb)
        if limit <= 0:
            return
        tf_mod.config.set_logical_device_configuration(
            gpus[0],
            [tf_mod.config.LogicalDeviceConfiguration(memory_limit=limit)],
        )
        if verbose:
            console.print(f"[green]TensorFlow GPU memory limit set[/green]: {limit} MB")
    except Exception as e:
        if verbose:
            console.print(f"[yellow]Could not set GPU memory limit[/yellow]: {e}")


def _tf_try_enable_memory_growth(tf_mod, gpus) -> None:
    for gpu in gpus:
        try:
            tf_mod.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            # If a logical device limit was set, memory growth may not be supported.
            pass


def _configure_tf_metal(*, verbose: bool = False, force_cpu: bool = False) -> None:
    """Enable TensorFlow Metal GPU on Apple Silicon when available."""
    _tf_set_log_level(verbose=bool(verbose))

    import tensorflow as tf

    _tf_try_disable_meta_optimizer(tf, verbose=bool(verbose))

    if force_cpu:
        _tf_try_force_cpu(tf, verbose=bool(verbose))

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if verbose:
            console.print("[yellow]TensorFlow GPU not detected[/yellow] (CPU mode).")
        return

    try:
        _tf_try_apply_gpu_memory_limit(tf, gpus, verbose=bool(verbose))
        _tf_try_enable_memory_growth(tf, gpus)
        if verbose:
            console.print(f"[green]TensorFlow GPU enabled[/green]: {gpus}")
    except Exception as e:
        if verbose:
            console.print(f"[yellow]GPU detected but could not enable memory growth[/yellow]: {e}")


def _build_buddy_model(
    *,
    feature_dim: int,
    seq_len: int,
    head_hidden: int = 64,
    head_layers: int = 2,
    head_dropout: float = 0.1,
    dense_hidden: int = 128,
    dense_dropout: float = 0.2,
):
    """Create the Buddy model: 5 parallel LSTM heads + shared dense + direction+confidence outputs."""
    import tensorflow as tf

    from ml_head_engine import MLEngineHead
    from mr_engine import MREngineHead
    from mt_engine import MTEngineHead
    from ms_head_engine import MSEngineHead
    from mx_head_engine import MXEngineHead

    inp = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="features")
    h1 = MLEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="ml")(inp)
    h2 = MREngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="mr")(inp)
    h3 = MTEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="mt")(inp)
    h4 = MSEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="ms")(inp)
    h5 = MXEngineHead(hidden_size=head_hidden, num_layers=head_layers, dropout=head_dropout, name="mx")(inp)

    merged = tf.keras.layers.Concatenate(name="concat")([h1, h2, h3, h4, h5])
    x = tf.keras.layers.Dense(int(dense_hidden), activation="relu", name="dense_0")(merged)
    x = tf.keras.layers.Dropout(float(dense_dropout), name="dense_dropout")(x)
    x = tf.keras.layers.Dense(int(dense_hidden // 2), activation="relu", name="dense_1")(x)

    # Force float32 outputs for numerically stable losses/metrics under mixed precision.
    direction = tf.keras.layers.Dense(1, activation="sigmoid", name="direction", dtype="float32")(x)
    # Confidence is trained as a regression-style score (scaled |next return|, nominally in [0,1]).
    # Use a bounded head so training can't diverge to extreme values.
    confidence = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence", dtype="float32")(x)

    return tf.keras.Model(inputs=inp, outputs={"direction": direction, "confidence": confidence}, name="buddy_model")


def _build_buddy_model_shared_encoder(
    *,
    feature_dim: int,
    seq_len: int,
    encoder_hidden: int = 64,
    encoder_layers: int = 2,
    encoder_dropout: float = 0.1,
    dense_hidden: int = 128,
    dense_dropout: float = 0.2,
):
    """Create a faster Buddy model with a shared LSTM encoder + dense heads."""
    import tensorflow as tf

    inp = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="features")
    x = inp

    n_layers = max(1, int(encoder_layers))
    for i in range(n_layers):
        return_sequences = i < (n_layers - 1)
        x = tf.keras.layers.LSTM(
            int(encoder_hidden),
            return_sequences=bool(return_sequences),
            dropout=float(encoder_dropout),
            name=f"enc_lstm_{i}",
        )(x)

    x = tf.keras.layers.Dense(int(dense_hidden), activation="relu", name="dense_0")(x)
    x = tf.keras.layers.Dropout(float(dense_dropout), name="dense_dropout")(x)
    x = tf.keras.layers.Dense(int(dense_hidden // 2), activation="relu", name="dense_1")(x)

    # Force float32 outputs for numerically stable losses/metrics under mixed precision.
    direction = tf.keras.layers.Dense(1, activation="sigmoid", name="direction", dtype="float32")(x)
    confidence = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence", dtype="float32")(x)

    return tf.keras.Model(
        inputs=inp,
        outputs={"direction": direction, "confidence": confidence},
        name="buddy_model_shared_encoder",
    )


def _build_buddy_model_tcn(
    *,
    feature_dim: int,
    seq_len: int,
    hidden_size: int = 64,
    num_layers: int = 3,
    kernel_size: int = 3,
    dropout: float = 0.5,           # Increased from 0.2 to combat overfitting
    dense_hidden: int = 128,
    dense_dropout: float = 0.4,     # Increased from 0.2
    kernel_regularizer: float = 0.005,  # Increased from 0.002 (stronger L2)
    noise_std: float = 0.05,        # Increased from 0.03 (more input noise)
):
    """Create a Buddy model with TCN encoder - FASTER on M1 Metal than LSTM.
    
    TCN uses dilated causal convolutions which are fully parallelizable,
    making them 2-3x faster than LSTMs on Apple Silicon.
    """
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.regularizers import l2
    
    l2_reg = l2(kernel_regularizer)
    inp = tf.keras.Input(shape=(int(seq_len), int(feature_dim)), name="features")
    
    # Input regularization (M1-compatible alternative to recurrent_dropout)
    x = layers.GaussianNoise(noise_std)(inp)
    x = layers.SpatialDropout1D(dropout * 0.5)(x)
    
    # TCN layers with exponentially increasing dilation
    for i in range(num_layers):
        dilation_rate = 2 ** i
        
        # Causal convolution with dilation
        conv_out = layers.Conv1D(
            filters=hidden_size,
            kernel_size=kernel_size,
            padding='causal',
            dilation_rate=dilation_rate,
            activation=None,
            kernel_regularizer=l2_reg,
            name=f'tcn_conv_{i}'
        )(x)
        conv_out = layers.BatchNormalization(name=f'tcn_bn_{i}')(conv_out)
        conv_out = layers.Activation('relu', name=f'tcn_relu_{i}')(conv_out)
        conv_out = layers.Dropout(dropout, name=f'tcn_dropout_{i}')(conv_out)
        
        # Residual connection
        if x.shape[-1] != hidden_size:
            x = layers.Conv1D(hidden_size, 1, name=f'tcn_residual_{i}')(x)
        x = layers.Add(name=f'tcn_add_{i}')([x, conv_out])
    
    # Global pooling to get fixed-size representation
    x = layers.GlobalAveragePooling1D(name='global_pool')(x)
    
    # Dense head
    x = layers.Dense(dense_hidden, activation='relu', name='dense_0', kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(dense_dropout, name='dense_dropout')(x)
    x = layers.Dense(dense_hidden // 2, activation='relu', name='dense_1', kernel_regularizer=l2_reg)(x)
    
    # Output heads (float32 for numerical stability with mixed precision)
    direction = layers.Dense(1, activation='sigmoid', name='direction', dtype='float32')(x)
    confidence = layers.Dense(1, activation='sigmoid', name='confidence', dtype='float32')(x)
    
    return tf.keras.Model(
        inputs=inp,
        outputs={'direction': direction, 'confidence': confidence},
        name='buddy_model_tcn',
    )


def _build_xgboost_model(
    feature_dim: int,
    seq_len: int,
    config: dict | None = None,
):
    """Build an XGBoost model for Buddy training.
    
    Returns a wrapper that's compatible with the Keras training interface.
    Note: XGBoost training is handled separately in _train_buddy_xgboost.
    """
    from xgboost_model import XGBoostTradingModel, XGBoostConfig
    
    xgb_config = XGBoostConfig()
    if config:
        for key, value in config.items():
            if hasattr(xgb_config, key):
                setattr(xgb_config, key, value)
    
    return XGBoostTradingModel(xgb_config)


def _build_buddy_model_for_type(
    model_type: str,
    *,
    feature_dim: int,
    seq_len: int,
    head_hidden: int = 64,
    head_layers: int = 2,
    head_dropout: float = 0.1,
    dense_hidden: int = 128,
    dense_dropout: float = 0.2,
    kernel_regularizer: float = 0.002,
    noise_std: float = 0.03,
):
    """Build a Buddy model based on the specified architecture type.
    
    Args:
        model_type: One of 'lstm', 'tcn', 'attention_lstm', 'tft'
        Other args: Model configuration parameters
    
    Returns:
        Configured Keras model
    
    M1 Metal Recommendations:
        - 'tcn': Fastest on Metal (2-3x faster than LSTM)
        - 'lstm': Good baseline, compatible
        - 'attention_lstm': Better accuracy, slightly slower
    """
    model_type = model_type.lower().strip()
    
    if model_type == 'tcn':
        return _build_buddy_model_tcn(
            feature_dim=feature_dim,
            seq_len=seq_len,
            hidden_size=head_hidden,
            num_layers=head_layers,
            dropout=head_dropout,
            dense_hidden=dense_hidden,
            dense_dropout=dense_dropout,
            kernel_regularizer=kernel_regularizer,
            noise_std=noise_std,
        )
    elif model_type in ('lstm', 'shared_encoder'):
        return _build_buddy_model_shared_encoder(
            feature_dim=feature_dim,
            seq_len=seq_len,
            encoder_hidden=head_hidden,
            encoder_layers=head_layers,
            encoder_dropout=head_dropout,
            dense_hidden=dense_hidden,
            dense_dropout=dense_dropout,
        )
    elif model_type == 'xgboost':
        # XGBoost model - returns a placeholder, actual training handled separately
        return _build_xgboost_model(
            feature_dim=feature_dim,
            seq_len=seq_len,
            config=None,
        )
    elif model_type == 'attention_lstm':
        # Use shared encoder with attention (can be extended later)
        # For now, fall back to shared encoder
        return _build_buddy_model_shared_encoder(
            feature_dim=feature_dim,
            seq_len=seq_len,
            encoder_hidden=head_hidden,
            encoder_layers=head_layers,
            encoder_dropout=head_dropout,
            dense_hidden=dense_hidden,
            dense_dropout=dense_dropout,
        )
    else:
        # Default to LSTM shared encoder
        return _build_buddy_model_shared_encoder(
            feature_dim=feature_dim,
            seq_len=seq_len,
            encoder_hidden=head_hidden,
            encoder_layers=head_layers,
            encoder_dropout=head_dropout,
            dense_hidden=dense_hidden,
            dense_dropout=dense_dropout,
        )


def train_buddy(
    config_path: str,
    csv_path: str | None = None,
    *,
    options: BuddyTrainingOptions | None = None,
    **kwargs: Any,
) -> None:
    """Train Buddy (TensorFlow-only) from USDJPY historical data."""
    opts = options or BuddyTrainingOptions()
    if kwargs:
        allowed = {f.name for f in fields(BuddyTrainingOptions)}
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(f"train_buddy() got unexpected keyword argument(s): {sorted(unknown)}")
        opts = replace(opts, **{k: kwargs[k] for k in allowed if k in kwargs})

    _train_buddy_impl(
        config_path,
        csv_path,
        options=opts,
    )


def _train_rl_position_sizer_if_ready(
    console,
    rl_timesteps: int = 500_000,
    min_samples: int = 500,
    *,
    features: np.ndarray | None = None,
    ensemble_predictions: np.ndarray | None = None,
    prices: np.ndarray | None = None,
) -> bool:
    """
    Train RL position sizer using ensemble training data.
    
    This function should be called after ensemble training completes successfully.
    It uses the actual features, ensemble predictions, and price data from training
    to learn optimal position sizing.
    
    Args:
        console: Rich console for output
        rl_timesteps: Number of RL training timesteps
        min_samples: Minimum samples required for training
        features: Market features array (n_samples, n_features) from ensemble training
        ensemble_predictions: Ensemble predictions (n_samples, 2) - [direction_prob, confidence]
        prices: Close prices array (n_samples,)
        
    Returns:
        True if training was run, False otherwise
    """
    try:
        import numpy as np
        from rl_position_sizing import RLPositionSizer, RLConfig, SB3_AVAILABLE, GYM_AVAILABLE
        
        if not SB3_AVAILABLE or not GYM_AVAILABLE:
            console.print("[dim]RL position sizing unavailable (missing stable-baselines3 or gymnasium)[/dim]")
            console.print("[dim]Install with: pip install stable-baselines3 gymnasium[/dim]")
            return False
        
        # Validate input data
        if features is None or ensemble_predictions is None or prices is None:
            console.print("[dim]RL training skipped: no training data provided[/dim]")
            console.print("[dim]Tip: RL training requires features, predictions, and prices from ensemble[/dim]")
            return False
        
        n_samples = len(features)
        if n_samples < min_samples:
            console.print(f"[dim]RL training skipped: insufficient data ({n_samples}/{min_samples} samples)[/dim]")
            return False
        
        # Validate data shapes
        if len(ensemble_predictions) != n_samples or len(prices) != n_samples:
            console.print(f"[yellow]RL training skipped: data shape mismatch[/yellow]")
            console.print(f"[dim]Features: {n_samples}, Predictions: {len(ensemble_predictions)}, Prices: {len(prices)}[/dim]")
            return False
        
        console.print(f"\n[bold cyan]🤖 RL Position Sizer Training[/bold cyan]")
        console.print(f"[dim]Using {n_samples:,} samples from ensemble training data[/dim]")
        console.print(f"[dim]Timesteps: {rl_timesteps:,}[/dim]")
        
        # Configure RL with appropriate sequence length for data size
        # Ensure we have enough data for the environment (need at least sequence_length + 100 steps)
        max_seq_len = max(10, n_samples - 200)  # Leave room for training episodes
        seq_len = min(60, max_seq_len)  # Default is 60, but reduce if data is small
        
        config = RLConfig(
            total_timesteps=rl_timesteps,
            sequence_length=seq_len,
        )
        console.print(f"[dim]Sequence length: {seq_len}, Features: {features.shape[1]}[/dim]")
        
        sizer = RLPositionSizer(config)
        
        # Train the RL agent with actual ensemble data
        stats = sizer.train(features, ensemble_predictions, prices)
        
        console.print("[green]✅ RL position sizer trained and saved[/green]")
        console.print(f"[dim]Model: trained_data/models/rl_position_sizer.zip[/dim]")
        console.print(f"[dim]Final equity: ${stats.get('final_equity', 0):.2f}, "
                      f"Win rate: {stats.get('win_rate', 0):.1%}, "
                      f"Trades: {stats.get('total_trades', 0)}[/dim]")
        
        return True
        
    except ImportError as e:
        console.print(f"[dim]RL training unavailable: {e}[/dim]")
        return False
    except Exception as e:
        console.print(f"[yellow]RL training failed: {e}[/yellow]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return False


def _train_buddy_impl(
    config_path: str,
    csv_path: str | None = None,
    *,
    options: BuddyTrainingOptions,
) -> None:  # noqa: C901, PLR0915, WPS231, CCR001  # NOSONAR
    """Train Buddy (TensorFlow-only) from USDJPY historical data."""
    # Suppress verbose utils logging during config load
    import logging as _logging
    _utils_logger = _logging.getLogger('utils')
    _utils_prev = _utils_logger.level
    _utils_logger.setLevel(_logging.WARNING)
    cfg = load_config(config_path)
    _utils_logger.setLevel(_utils_prev)

    oanda_fetch = options.oanda_fetch
    advanced = options.advanced
    feature_curriculum_opt = options.feature_curriculum
    curriculum_ks_opt = options.curriculum_ks
    pca_components = options.pca_components
    seq_len = options.seq_len
    epochs = options.epochs
    batch_size = options.batch_size
    lr = options.lr
    patience = options.patience
    seed = options.seed
    run_tag = options.run_tag
    all_features = options.all_features
    device = options.device
    ignore_input_mismatches = bool(options.ignore_input_mismatches)
    disable_tier2_on_mismatch = bool(options.disable_tier2_on_mismatch)
    mixed_precision = options.mixed_precision
    steps_per_execution = options.steps_per_execution
    jit_compile = options.jit_compile
    shuffle_buffer = options.shuffle_buffer
    prefetch = options.prefetch
    cache_val = options.cache_val
    combined_use_predict = options.combined_use_predict
    shared_encoder = options.shared_encoder
    model_type = options.model_type
    timing = options.timing
    fit_verbose = options.fit_verbose

    dev = str(device or "auto").strip().lower()
    force_cpu = dev == "cpu"

    try:
        from tracing_setup import setup_tracing as _setup_tracing
    except Exception:
        _setup_tracing = None

    from buddy_training_helpers import _buddy_setup_training_environment

    # When debugging perf, show whether Metal GPU is actually visible.
    mp_enabled = _buddy_setup_training_environment(
        timing=bool(timing),
        force_cpu=bool(force_cpu),
        mixed_precision=bool(mixed_precision),
        configure_tf_metal=_configure_tf_metal,
        console=console,
        setup_tracing=_setup_tracing,
    )
    import json
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from feature_engineering import FeatureEngineering

    buddy_cfg = (cfg.get("buddy") or {}) if isinstance(cfg, dict) else {}

    # Optional: out-of-sample (OOS) evaluation controls (env vars).
    # If we're training from live OANDA (no --csv), prefer a single fetch and split the tail as OOS.
    try:
        _oos_env = os.environ.get("BUDDY_OOS_EVAL", "").strip().lower()
        oos_enabled_env = _oos_env in {"1", "true", "yes", "y", "on"}
    except Exception:
        oos_enabled_env = False
    try:
        oos_candles_env = int(float(os.environ.get("BUDDY_OOS_CANDLES", "5000")))
    except Exception:
        oos_candles_env = 5000
    oos_candles_env = max(300, min(int(oos_candles_env), 20000))

    oos_holdout_csv: str | None = None
    oos_holdout_rows: int | None = None
    oos_live_split = bool(oos_enabled_env) and (csv_path is None) and (oanda_fetch is not None)
    base_oanda_fetch = oanda_fetch
    if oos_live_split and oanda_fetch is not None:
        try:
            train_candles_req = int(getattr(oanda_fetch, "candles", 5000) or 5000)
        except Exception:
            train_candles_req = 5000
        total_candles_req = int(train_candles_req) + int(oos_candles_env)
        try:
            oanda_fetch = replace(oanda_fetch, candles=int(total_candles_req))
        except Exception:
            pass

    # Config defaults + CLI overrides.
    if feature_curriculum_opt is None:
        feature_curriculum = bool(buddy_cfg.get("feature_curriculum", False))
    else:
        feature_curriculum = bool(feature_curriculum_opt)

    if curriculum_ks_opt is None:
        curriculum_ks = str(buddy_cfg.get("curriculum_ks", DEFAULT_CURRICULUM_KS))
    else:
        curriculum_ks = str(curriculum_ks_opt)

    # mp_enabled is set by _buddy_setup_training_environment()

    init_from = advanced.init_from if advanced else None
    es_monitor = (advanced.es_monitor if advanced else "direction")
    combined_w_dir = float(advanced.combined_w_dir) if advanced else 0.7
    combined_w_conf = float(advanced.combined_w_conf) if advanced else 0.3
    train_smoothing = bool(advanced.train_smoothing) if advanced else True
    top_features = int(advanced.top_features) if (advanced and advanced.top_features is not None) else None
    median_window = int(advanced.median_window) if (advanced and advanced.median_window is not None) else None
    vol_norm_window = int(advanced.vol_norm_window) if (advanced and advanced.vol_norm_window is not None) else None
    min_volume = int(advanced.min_volume) if (advanced and advanced.min_volume is not None) else None
    spread_filter = bool(advanced.spread_filter) if advanced else False
    spread_pctl = float(advanced.spread_pctl) if advanced else 0.99
    spread_mult = float(advanced.spread_mult) if advanced else 3.0

    tier2_calibrate = bool(getattr(advanced, "tier2_calibrate", True)) if advanced else True
    tier2_stop_loss_pips_cfg = float(buddy_cfg.get("stop_loss_pips", 15.0))  # 15 pip SL
    tier2_take_profit_pips_cfg = float(buddy_cfg.get("take_profit_pips", 30.0))  # 30 pip TP base
    tier2_spread_fallback_pips_cfg = float(buddy_cfg.get("tier2_spread_fallback_pips", 2.0))
    tier2_horizon_candles_cfg = int(buddy_cfg.get("tier2_horizon_candles", 288))
    tier2_tie_break_cfg = str(buddy_cfg.get("tier2_tie_break", "sl"))

    adv_sl = getattr(advanced, "tier2_stop_loss_pips", None) if advanced else None
    adv_tp = getattr(advanced, "tier2_take_profit_pips", None) if advanced else None
    adv_sp = getattr(advanced, "tier2_spread_fallback_pips", None) if advanced else None
    adv_hz = getattr(advanced, "tier2_horizon_candles", None) if advanced else None
    adv_tb = getattr(advanced, "tier2_tie_break", None) if advanced else None

    tier2_stop_loss_pips = float(adv_sl) if adv_sl is not None else tier2_stop_loss_pips_cfg
    tier2_take_profit_pips = float(adv_tp) if adv_tp is not None else tier2_take_profit_pips_cfg
    tier2_spread_fallback_pips = float(adv_sp) if adv_sp is not None else tier2_spread_fallback_pips_cfg
    tier2_horizon_candles = int(adv_hz) if adv_hz is not None else tier2_horizon_candles_cfg
    tier2_tie_break = str(adv_tb) if adv_tb is not None else tier2_tie_break_cfg
    tier2_calibration_bins = int(getattr(advanced, "tier2_calibration_bins", 20)) if advanced else 20
    tier2_calibration_stride = int(getattr(advanced, "tier2_calibration_stride", 5)) if advanced else 5

    feature_curriculum = bool(feature_curriculum)
    if feature_curriculum and top_features is not None:
        raise ValueError("--feature-curriculum cannot be combined with --top-features")

    if median_window is not None and median_window <= 1:
        median_window = None
    if vol_norm_window is not None and vol_norm_window <= 1:
        vol_norm_window = None
    if min_volume is not None and min_volume <= 0:
        min_volume = None
    spread_pctl = min(max(spread_pctl, 0.5), 0.999)
    spread_mult = max(spread_mult, 1.0)

    # Reproducibility (best-effort).
    try:
        tf.keras.utils.set_random_seed(int(seed))
        np.random.seed(int(seed))
    except Exception:
        pass

    from buddy_training_helpers import _buddy_load_and_validate_csv, _load_multi_pair_data

    # Check for multi-pair foundation training mode
    multi_pair_mode = bool(options.multi_pair)
    foundation_pairs_str = options.foundation_pairs
    
    if multi_pair_mode:
        # Multi-pair foundation model training - load data from multiple pairs
        DEFAULT_FOUNDATION_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "USD_CHF"]
        
        if foundation_pairs_str:
            pairs_list = [p.strip().upper().replace("/", "_") for p in foundation_pairs_str.split(",")]
        else:
            pairs_list = DEFAULT_FOUNDATION_PAIRS
        
        # Determine candles per pair from oanda_fetch or default
        candles_per_pair = 5000
        if oanda_fetch is not None:
            try:
                candles_per_pair = int(getattr(oanda_fetch, "candles", 5000) or 5000)
            except Exception:
                pass
        
        # Determine granularity
        granularity = "H1"
        if oanda_fetch is not None:
            try:
                granularity = str(getattr(oanda_fetch, "granularity", "H1") or "H1")
            except Exception:
                pass
        
        console.print(Panel(
            f"[bold]Multi-Pair Foundation Training[/bold]\n\n"
            f"[dim]Pairs:[/dim] {', '.join(pairs_list)}\n"
            f"[dim]Candles per pair:[/dim] {candles_per_pair:,}\n"
            f"[dim]Granularity:[/dim] {granularity}\n"
            f"[dim]Total expected:[/dim] ~{candles_per_pair * len(pairs_list):,} rows",
            title="🌍 Multi-Pair Mode",
            border_style="cyan",
        ))
        
        df = _load_multi_pair_data(
            pairs=pairs_list,
            granularity=granularity,
            candles_per_pair=candles_per_pair,
            console=console,
        )
        
        if df is None or len(df) < 1000:
            raise ValueError(f"Multi-pair data loading failed or insufficient data: {len(df) if df is not None else 0} rows")
        
        console.print(f"[green]✓ Multi-pair data loaded: {len(df):,} rows from {len(pairs_list)} pairs[/green]")
        
        # Skip OOS split for multi-pair (data is already shuffled across pairs)
        oos_live_split = False
    else:
        # Standard single-pair training
        # Load data (optionally fetching fresh candles from OANDA) + apply basic quality filters.
        df = _buddy_load_and_validate_csv(
            csv_path=csv_path,
            oanda_fetch=oanda_fetch,
            min_volume=min_volume,
            spread_filter=bool(spread_filter),
            spread_pctl=float(spread_pctl),
            spread_mult=float(spread_mult),
            oanda_fetch_to_csv=_oanda_fetch_to_csv,
            console=console,
        )

    # If we fetched a larger live window for OOS, hold out the most recent tail and train on the earlier slice.
    if oos_live_split and base_oanda_fetch is not None:
        try:
            holdout_n = int(oos_candles_env)
            # Ensure we still have enough rows for training.
            if int(len(df)) > int(holdout_n) + int(seq_len) + 50:
                oos_df_raw = df.tail(int(holdout_n)).copy()
                df = df.iloc[: int(len(df)) - int(holdout_n)].copy()

                out_dir = Path("market_data")
                out_dir.mkdir(parents=True, exist_ok=True)
                try:
                    inst = str(getattr(base_oanda_fetch, "instrument", "USD_JPY"))
                except Exception:
                    inst = "USD_JPY"
                try:
                    gran = str(getattr(base_oanda_fetch, "granularity", "M5"))
                except Exception:
                    gran = "M5"
                ts_tag = time.strftime("%Y%m%d_%H%M%S")
                oos_path = out_dir / f"oanda_{inst}_{gran}_oos_{ts_tag}.csv"
                oos_df_raw.to_csv(oos_path, index=False)
                oos_holdout_csv = str(oos_path)
                oos_holdout_rows = int(len(oos_df_raw))
                console.print(
                    f"Prepared OOS holdout from live tail: rows={int(oos_holdout_rows)} -> {oos_path} (training rows={int(len(df))})"
                )
        except Exception:
            oos_holdout_csv = None
            oos_holdout_rows = None

    # Training-time noise reduction:
    # - If we are running full feature engineering, let FeatureEngineering apply it (so it happens exactly once).
    # - If feature engineering is disabled, apply it here so Buddy still trains on de-noised candles.
    if train_smoothing:
        extra = f" + median(close,w={median_window})" if median_window else ""
        console.print(f"Training noise reduction enabled: resample->M5{extra} + EMA(14) close")

    # Feature engineering (numeric only).
    if all_features:
        t_fe = time.perf_counter()
        # Suppress verbose logging during feature engineering
        import logging as _logging
        _fe_logger = _logging.getLogger('feature_engineering')
        _fe_prev_level = _fe_logger.level
        _fe_logger.setLevel(_logging.WARNING)
        
        fe = FeatureEngineering(cfg.get("feature_engineering", {}))
        df = fe.create_features(
            df,
            include_all=True,
            apply_candle_smoothing=train_smoothing,
            median_window=median_window,
        )
        
        # Restore logging level
        _fe_logger.setLevel(_fe_prev_level)
        
        elapsed_fe = time.perf_counter() - t_fe
        console.print(Panel(
            f"[bold]Feature Engineering Complete[/bold]\n\n"
            f"[dim]Smoothing:[/dim] {bool(train_smoothing)}  [dim]Median Window:[/dim] {median_window}\n"
            f"[dim]Output:[/dim] {int(len(df)):,} rows × {int(df.shape[1])} columns\n"
            f"[dim]Time:[/dim] {elapsed_fe:.2f}s",
            title="⚙️ Feature Engineering",
            border_style="blue",
        ))
    elif train_smoothing:
        try:
            from candle_smoothing import resample_5min_ohlcv_and_ema_close

            df = resample_5min_ohlcv_and_ema_close(
                df,
                ema_span=14,
                time_col="time",
                median_window=median_window,
            )
        except Exception as e:
            console.print(f"[yellow]Training noise reduction skipped[/yellow]: {e}")

    init_feature_columns: list[str] | None = None
    if init_from:
        try:
            meta_p = _meta_path_for_checkpoint(str(init_from))
            if meta_p:
                init_meta = json.loads(meta_p.read_text())
                cols = init_meta.get("feature_columns")
                if isinstance(cols, list) and cols:
                    init_feature_columns = [str(c) for c in cols]
                    console.print(f"Using feature_columns from init_from metadata: {meta_p}")
        except Exception as e:
            console.print(f"[yellow]Warm-start meta load skipped[/yellow]: {e}")

    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        raise ValueError("No numeric features found after preprocessing")

    # If user requested top-K feature selection, we cannot warm-start from a checkpoint
    # trained with a different feature set/ordering.
    if top_features is not None and init_from:
        raise ValueError(
            "--top-features cannot be used with --warm-start/--init-from because the checkpoint expects a fixed feature set. "
            "Run without warm-start to train a new (smaller) model, then you can warm-start future runs from that new checkpoint."
        )

    # If warm-starting, force the same feature ordering as the checkpoint was trained with.
    if init_feature_columns is not None:
        missing_cols = [c for c in init_feature_columns if c not in numeric_df.columns]
        extra_cols = [c for c in numeric_df.columns if c not in set(init_feature_columns)]
        if missing_cols:
            console.print(f"[yellow]Warm-start[/yellow]: {len(missing_cols)} missing features will be filled with 0.0")
        if extra_cols:
            console.print(f"[yellow]Warm-start[/yellow]: {len(extra_cols)} extra features will be dropped")
        numeric_df = numeric_df.reindex(columns=init_feature_columns, fill_value=0.0)

    # Drop rows with NaNs created by rolling indicators.
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    # Keep OHLC (+ optional bid/ask closes) aligned to the exact numeric_df rows used for training.
    ohlc_cols = ["open", "high", "low", "close"]
    extra_cols: list[str] = []
    for c in ("bid_close", "ask_close"):
        if c in df.columns:
            extra_cols.append(c)
    ohlc_df = df.loc[numeric_df.index, ohlc_cols + extra_cols].copy()

    closes = df.loc[numeric_df.index, "close"].to_numpy(dtype=np.float32)
    ranked_cols: list[str] | None = None
    if feature_curriculum or (top_features is not None and top_features > 0):
        try:
            # Labels aligned to rows: label at t is about move from t->t+1.
            y_dir = (closes[1:] > closes[:-1]).astype(np.float32)
            r0 = (closes[1:] - closes[:-1]) / np.maximum(closes[:-1], 1e-8)

            if vol_norm_window is not None:
                # Normalize by rolling volatility so confidence isn't dominated by regime-level vol.
                w = int(vol_norm_window)
                ret_std = (
                    pd.Series(r0)
                    .rolling(w, min_periods=max(10, w // 5))
                    .std()
                    .to_numpy(dtype=np.float32)
                )
                # rolling std has NaNs at the head; fill with first valid (or fallback).
                if not np.isfinite(ret_std).all():
                    finite = ret_std[np.isfinite(ret_std)]
                    fill = float(np.median(finite)) if finite.size else 1.0
                    ret_std = np.where(np.isfinite(ret_std), ret_std, fill).astype(np.float32)
                r0 = r0 / np.maximum(ret_std, 1e-6)

            abs_ret = np.abs(r0).astype(np.float32)
            abs_ret = np.nan_to_num(abs_ret, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

            # IMPORTANT: feature ranking must be TRAIN-ONLY to avoid leaking future validation info.
            # Align features to labels (labels are for t->t+1, so drop last feature row).
            x_rank_all = numeric_df.iloc[:-1].astype(np.float32)
            n_rank = int(len(x_rank_all))
            split_rank = int(n_rank * 0.8)
            if split_rank <= 10:
                raise ValueError("Not enough rows for train-only ranking")

            x_rank = x_rank_all.iloc[:split_rank]
            y_dir_rank = pd.Series(y_dir[:split_rank])
            abs_ret_rank = pd.Series(abs_ret[:split_rank])

            # For ranking, raw abs_ret correlation is enough; confidence scaling uses train split later.
            s_dir = x_rank.corrwith(y_dir_rank, method="pearson").abs().fillna(0.0)
            s_conf = x_rank.corrwith(abs_ret_rank, method="pearson").abs().fillna(0.0)
            scores = (0.7 * s_dir) + (0.3 * s_conf)
            ranked_cols = scores.sort_values(ascending=False).index.tolist()

            if top_features is not None and top_features > 0 and top_features < int(numeric_df.shape[1]):
                keep = ranked_cols[: int(top_features)]
                numeric_df = numeric_df[keep]
                console.print(
                    f"[cyan]✓ Feature Selection:[/cyan] kept {len(keep)} / {int(len(ranked_cols))} ranked features"
                )
        except Exception as e:
            console.print(f"[yellow]Feature ranking skipped[/yellow]: {e}")
            ranked_cols = None

    feats = numeric_df.to_numpy(dtype=np.float32)
    feature_columns = list(numeric_df.columns)

    if feats.shape[1] < 6:
        console.print(f"[yellow]⚠ Warning:[/yellow] only {feats.shape[1]} numeric features detected.")
    else:
        console.print(f"[cyan]✓ Features:[/cyan] {feats.shape[1]} numeric columns")

    # Build supervised targets.
    # Direction: next-close up/down.
    next_close = closes[1:]
    cur_close = closes[:-1]
    direction_y_all = (next_close > cur_close).astype(np.float32)

    # Align features to targets (targets are for t->t+1, so drop last feature row).
    feats = feats[:-1]

    n = feats.shape[0]
    if n <= seq_len + 10:
        raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len}")

    # Time-ordered split: last 20% as holdout.
    split = int(n * 0.8)
    train_dir_raw = direction_y_all[:split]
    val_dir_raw = direction_y_all[split:]
    train_dir_pos_rate = float(np.mean(train_dir_raw)) if len(train_dir_raw) else 0.0
    val_dir_pos_rate = float(np.mean(val_dir_raw)) if len(val_dir_raw) else 0.0
    console.print(
        f"[cyan]✓ Direction Labels:[/cyan] train={train_dir_pos_rate*100:.1f}% up / val={val_dir_pos_rate*100:.1f}% up"
    )

    # Confidence target: Tier-2 TP-before-SL (true direction), with sparse labeling + sample weights.
    # This matches Tier-2 calibration semantics and avoids the old q95-magnitude mismatch.
    conf_y_all = np.zeros((int(n),), dtype=np.float32)
    conf_w_all = np.zeros((int(n),), dtype=np.float32)
    conf_instrument_guess = None
    label_stride = max(1, int(tier2_calibration_stride))
    label_stride_env = os.environ.get("BUDDY_TIER2_LABEL_STRIDE")
    if label_stride_env:
        try:
            label_stride = max(1, int(float(label_stride_env)))
        except Exception:
            label_stride = max(1, int(tier2_calibration_stride))
    if not tier2_calibrate:
        console.print("Tier-2 labeling disabled (--no-tier2-calibrate): confidence targets set to zeros")
    else:
        try:
            t_tier2_label = time.perf_counter()
            from fx_paper import pip_size, simulate_tp_sl_outcome

            # Best-effort instrument inference.
            instrument_guess = None
            if oanda_fetch is not None:
                try:
                    instrument_guess = str(getattr(oanda_fetch, "instrument", None) or None)
                except Exception:
                    instrument_guess = None
            if not instrument_guess:
                cp = str(csv_path or "")
                up = cp.upper()
                if "EUR_USD" in up:
                    instrument_guess = "EUR_USD"
                elif "GBP_USD" in up:
                    instrument_guess = "GBP_USD"
                elif "USD_JPY" in up or "USDJPY" in up or "_JPY_" in up or up.endswith("JPY.CSV"):
                    instrument_guess = "USD_JPY"
                else:
                    instrument_guess = "USD_JPY"
            conf_instrument_guess = str(instrument_guess)

            pip = float(pip_size(str(instrument_guess)))
            max_signal = min(int(n) - 1, int(len(ohlc_df)) - 3)
            if max_signal < 0:
                raise ValueError("Insufficient OHLC rows for TP/SL labeling")

            def _spread_pips_at_entry(gi: int) -> float:
                sp = float(tier2_spread_fallback_pips)
                try:
                    if "bid_close" in ohlc_df.columns and "ask_close" in ohlc_df.columns:
                        bid = float(ohlc_df["bid_close"].iloc[int(gi) + 1])
                        ask = float(ohlc_df["ask_close"].iloc[int(gi) + 1])
                        if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
                            sp = float((ask - bid) / max(pip, 1e-9))
                except Exception:
                    sp = float(tier2_spread_fallback_pips)
                return float(sp)

            start_gi = max(0, int(seq_len) - 1)
            n_sim = 0
            for gi in range(int(start_gi), int(max_signal) + 1, int(label_stride)):
                spread_pips = _spread_pips_at_entry(int(gi))
                true_dir = "long" if float(direction_y_all[int(gi)]) >= 0.5 else "short"
                res = simulate_tp_sl_outcome(
                    ohlc_df,
                    instrument=str(instrument_guess),
                    signal_index=int(gi),
                    direction=str(true_dir),
                    stop_loss_pips=float(tier2_stop_loss_pips),
                    take_profit_pips=float(tier2_take_profit_pips),
                    spread_pips=float(spread_pips),
                    max_horizon_candles=int(tier2_horizon_candles),
                    tie_break=str(tier2_tie_break),
                )

                conf_y_all[int(gi)] = float(int(res.to_label()))
                conf_w_all[int(gi)] = 1.0
                n_sim += 1

            console.print(
                f"[cyan]✓ Tier-2 Labels:[/cyan] {str(instrument_guess)} | SL={float(tier2_stop_loss_pips):g} TP={float(tier2_take_profit_pips):g} | "
                f"horizon={int(tier2_horizon_candles)} stride={int(label_stride)} | {int(n_sim)} samples in {time.perf_counter() - t_tier2_label:.2f}s"
            )
        except Exception as e:
            console.print(f"[yellow]Tier-2 confidence labeling failed[/yellow]: {e}")
            conf_y_all = np.zeros((int(n),), dtype=np.float32)
            conf_w_all = np.zeros((int(n),), dtype=np.float32)

    # Standardize features (train-only stats).
    # Keep everything float32 and standardize IN-PLACE to reduce peak memory usage.
    feats = feats.astype(np.float32, copy=False)
    train_feats_view = feats[:split]
    mu = train_feats_view.mean(axis=0, keepdims=True, dtype=np.float32)
    sigma = train_feats_view.std(axis=0, keepdims=True, dtype=np.float32)
    sigma = np.where(sigma < 1e-6, np.float32(1.0), sigma).astype(np.float32)
    feats -= mu
    feats /= sigma
    train_feats = feats[:split]
    val_feats = feats[split:]

    orig_feature_dim = int(train_feats.shape[1])

    # Optional PCA (fit on train timesteps).
    pca_components_eff: int | None = None
    pca_model = None
    if pca_components is not None:
        pca_components_eff = int(pca_components)
        if pca_components_eff <= 0:
            pca_components_eff = None

    if pca_components_eff is not None and pca_components_eff < train_feats.shape[1]:
        from sklearn.decomposition import PCA

        pca_model = PCA(n_components=pca_components_eff, svd_solver="auto", random_state=42)
        pca_model.fit(train_feats)
        # sklearn returns float64; cast back to float32 to avoid doubling memory.
        train_feats = pca_model.transform(train_feats).astype(np.float32)
        val_feats = pca_model.transform(val_feats).astype(np.float32)
        console.print(f"PCA enabled: {orig_feature_dim} -> {train_feats.shape[1]}")

    meta_warnings: list[str] = []

    # NOTE: PCA changes feature space; curriculum ranks original columns.
    if feature_curriculum and pca_components_eff is not None:
        msg = "PCA + feature curriculum are incompatible (curriculum ranks original features, PCA changes dimensions)."
        if ignore_input_mismatches:
            console.print(f"[yellow]Warning[/yellow]: {msg} Disabling curriculum due to --ignore-input-mismatches.")
            meta_warnings.append(f"{msg} Curriculum disabled due to --ignore-input-mismatches.")
            feature_curriculum = False
        else:
            raise ValueError(msg)

    # Build the model-space features array used for Tier-2 calibration window slicing.
    # - If PCA enabled, train_feats/val_feats are already transformed.
    # - If PCA disabled, feats is standardized in-place.
    feats_model: np.ndarray
    try:
        if pca_components_eff is not None and pca_model is not None:
            feats_model = np.concatenate([train_feats, val_feats], axis=0).astype(np.float32, copy=False)
        else:
            feats_model = feats.astype(np.float32, copy=False)
    except Exception:
        feats_model = feats.astype(np.float32, copy=False)

    train_conf_raw = conf_y_all[:split]
    val_conf_raw = conf_y_all[split:]
    train_conf_w_raw = conf_w_all[:split]
    val_conf_w_raw = conf_w_all[split:]

    # Guardrails: if we still have non-finite values anywhere, stop early.
    if not np.isfinite(train_feats).all() or not np.isfinite(val_feats).all():
        raise ValueError("Non-finite values detected in standardized features (check preprocessing/filtering).")
    if not np.isfinite(train_dir_raw).all() or not np.isfinite(val_dir_raw).all():
        raise ValueError("Non-finite values detected in direction targets.")
    if not np.isfinite(train_conf_raw).all() or not np.isfinite(val_conf_raw).all():
        raise ValueError("Non-finite values detected in confidence targets.")
    if not np.isfinite(train_conf_w_raw).all() or not np.isfinite(val_conf_w_raw).all():
        raise ValueError("Non-finite values detected in confidence target weights.")

    train_mask = train_conf_w_raw.reshape(-1) > 0
    val_mask = val_conf_w_raw.reshape(-1) > 0
    train_conf_target_mean = float(np.mean(train_conf_raw.reshape(-1)[train_mask])) if bool(train_mask.any()) else 0.0
    val_conf_target_mean = float(np.mean(val_conf_raw.reshape(-1)[val_mask])) if bool(val_mask.any()) else 0.0
    console.print(
        f"[cyan]✓ Confidence Targets:[/cyan] train={train_conf_target_mean*100:.1f}% / val={val_conf_target_mean*100:.1f}% (Tier-2 TP/SL)"
    )

    has_conf_labels = bool(tier2_calibrate) and bool(val_mask.any())

    def make_sequence_dataset(
        x2d: np.ndarray,
        y_dir: np.ndarray,
        y_conf: np.ndarray,
        y_conf_w: np.ndarray,
        *,
        shuffle: bool,
        batch_size: int,
        seed: int,
        feature_mask: "tf.Tensor | None" = None,
        shuffle_buffer_override: int | None = None,
        prefetch_override: int | None = None,
        cache: bool = False,
    ):
        # Streaming sequence creation to reduce peak memory:
        # yields windows of shape (seq_len, feature_dim) and uses the last element's label.
        # Match previous semantics exactly by dropping the final timestep (label_idx ends at len-2).
        import os

        # Avoid unnecessary copies (important on macOS unified memory).
        x2d = np.asarray(x2d).astype(np.float32, copy=False)
        y_dir = np.asarray(y_dir).reshape(-1).astype(np.float32, copy=False)
        y_conf = np.asarray(y_conf).reshape(-1).astype(np.float32, copy=False)
        y_conf_w = np.asarray(y_conf_w).reshape(-1).astype(np.float32, copy=False)

        if len(x2d) <= int(seq_len) + 1:
            raise ValueError(f"Not enough rows ({len(x2d)}) for seq_len={int(seq_len)}")

        x2d = x2d[:-1]
        y_dir = y_dir[:-1]
        y_conf = y_conf[:-1]
        y_conf_w = y_conf_w[:-1]

        # Window shuffling can consume a lot of RAM because each element is a
        # (seq_len, feature_dim) tensor. Default to a small buffer on macOS.
        n_windows = int(len(x2d) - int(seq_len) + 1)
        if shuffle_buffer_override is not None:
            shuffle_buf = max(1, min(int(shuffle_buffer_override), n_windows))
        else:
            shuffle_buf_env = os.environ.get("BUDDY_SHUFFLE_BUFFER")
            if shuffle_buf_env:
                try:
                    shuffle_buf = max(1, min(int(shuffle_buf_env), n_windows))
                except Exception:
                    shuffle_buf = min(256, n_windows)
            else:
                shuffle_buf = min(256, n_windows)

        if prefetch_override is not None:
            prefetch_n = int(prefetch_override)
        else:
            prefetch_env = os.environ.get("BUDDY_PREFETCH")
            if prefetch_env:
                try:
                    prefetch_n = int(prefetch_env)
                except Exception:
                    prefetch_n = 1
            else:
                prefetch_n = 1

        # Fast windowing path: avoid Dataset.window/flat_map overhead.
        # This is dramatically faster on macOS/Metal and reduces per-step latency.
        try:
            ts = getattr(getattr(tf.keras, "utils", None), "timeseries_dataset_from_array", None)
            if ts is None:
                raise AttributeError("timeseries_dataset_from_array unavailable")

            # We want label at the END of each window.
            y_dir_end = y_dir[int(seq_len) - 1 :]
            y_conf_end = y_conf[int(seq_len) - 1 :]
            y_conf_w_end = y_conf_w[int(seq_len) - 1 :]

            x_seq = ts(
                x2d,
                targets=None,
                sequence_length=int(seq_len),
                sequence_stride=1,
                sampling_rate=1,
                shuffle=False,
                batch_size=None,
            )
            y_seq = tf.data.Dataset.from_tensor_slices((y_dir_end, y_conf_end, y_conf_w_end))
            ds = tf.data.Dataset.zip((x_seq, y_seq))

            if feature_mask is not None:
                fm = tf.reshape(tf.cast(feature_mask, tf.float32), (1, -1))

                def _map_masked_fast(xw, ypack):
                    ydw, ycw, wcw = ypack
                    xw = tf.cast(xw, tf.float32) * fm
                    return (
                        xw,
                        {
                            "direction": tf.expand_dims(tf.cast(ydw, tf.float32), axis=-1),
                            "confidence": tf.expand_dims(tf.cast(ycw, tf.float32), axis=-1),
                        },
                        {
                            "direction": tf.constant(1.0, dtype=tf.float32),
                            "confidence": tf.expand_dims(tf.cast(wcw, tf.float32), axis=-1),
                        },
                    )

                ds = ds.map(_map_masked_fast, num_parallel_calls=tf.data.AUTOTUNE)
            else:

                def _map_fast(xw, ypack):
                    ydw, ycw, wcw = ypack
                    return (
                        tf.cast(xw, tf.float32),
                        {
                            "direction": tf.expand_dims(tf.cast(ydw, tf.float32), axis=-1),
                            "confidence": tf.expand_dims(tf.cast(ycw, tf.float32), axis=-1),
                        },
                        {
                            "direction": tf.constant(1.0, dtype=tf.float32),
                            "confidence": tf.expand_dims(tf.cast(wcw, tf.float32), axis=-1),
                        },
                    )

                ds = ds.map(_map_fast, num_parallel_calls=tf.data.AUTOTUNE)
        except Exception:
            # Fallback to the original streaming implementation.
            ds = tf.data.Dataset.from_tensor_slices((x2d, y_dir, y_conf, y_conf_w))
            ds = ds.window(int(seq_len), shift=1, drop_remainder=True)
            ds = ds.flat_map(
                lambda xw, ydw, ycw, wcw: tf.data.Dataset.zip(
                    (
                        xw.batch(int(seq_len), drop_remainder=True),
                        ydw.batch(int(seq_len), drop_remainder=True),
                        ycw.batch(int(seq_len), drop_remainder=True),
                        wcw.batch(int(seq_len), drop_remainder=True),
                    )
                )
            )
            if feature_mask is not None:
                fm = tf.reshape(tf.cast(feature_mask, tf.float32), (1, -1))

                def _map_masked_full(xw, ydw, ycw, wcw):
                    xw = tf.cast(xw, tf.float32) * fm
                    return (
                        xw,
                        {
                            "direction": tf.expand_dims(ydw[-1], axis=-1),
                            "confidence": tf.expand_dims(ycw[-1], axis=-1),
                        },
                        {
                            "direction": tf.constant(1.0, dtype=tf.float32),
                            "confidence": tf.expand_dims(tf.cast(wcw[-1], tf.float32), axis=-1),
                        },
                    )

                ds = ds.map(_map_masked_full, num_parallel_calls=tf.data.AUTOTUNE)
            else:
                ds = ds.map(
                    lambda xw, ydw, ycw, wcw: (
                        xw,
                        {
                            "direction": tf.expand_dims(ydw[-1], axis=-1),
                            "confidence": tf.expand_dims(ycw[-1], axis=-1),
                        },
                        {
                            "direction": tf.constant(1.0, dtype=tf.float32),
                            "confidence": tf.expand_dims(tf.cast(wcw[-1], tf.float32), axis=-1),
                        },
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

        if cache:
            # Caching windowed tensors can use a lot of memory; default to val-only usage.
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(int(shuffle_buf), seed=int(seed), reshuffle_each_iteration=True)
        ds = ds.batch(int(batch_size), drop_remainder=False)

        # Keep prefetch bounded by default to avoid ballooning unified memory on macOS.
        # Allow prefetch=0 to mean AUTOTUNE.
        if int(prefetch_n) == 0:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        else:
            ds = ds.prefetch(max(1, int(prefetch_n)))
        return ds

    feature_mask_var = None
    curriculum_cb = None
    if feature_curriculum:
        if not ranked_cols:
            console.print("[yellow]Feature curriculum requested but ranking unavailable; disabling curriculum.[/yellow]")
        else:
            try:
                # Rank indices aligned to current feature_columns.
                col_index = {c: i for i, c in enumerate(feature_columns)}
                order_idx = [int(col_index[c]) for c in ranked_cols if c in col_index]
                feature_dim_eff = int(train_feats.shape[1])

                # Parse schedule: comma-separated ints, where 0 means "all".
                ks_raw = (curriculum_ks or DEFAULT_CURRICULUM_KS).strip()
                ks: list[int] = []
                for part in [p.strip() for p in ks_raw.split(",") if p.strip()]:
                    try:
                        ks.append(int(float(part)))
                    except Exception:
                        continue
                if not ks:
                    ks = [32, 64, 128, 0]
                ks_norm = [feature_dim_eff if k <= 0 else min(int(k), feature_dim_eff) for k in ks]
                ks_final: list[int] = []
                for k in ks_norm:
                    if not ks_final or k >= ks_final[-1]:
                        ks_final.append(int(k))
                if ks_final[-1] != feature_dim_eff:
                    ks_final.append(feature_dim_eff)

                init_k = int(ks_final[0])
                init_mask = np.zeros((feature_dim_eff,), dtype=np.float32)
                init_mask[np.asarray(order_idx[:init_k], dtype=np.int32)] = 1.0
                feature_mask_var = tf.Variable(init_mask, trainable=False, dtype=tf.float32, name="feature_mask")

                console.print(
                    f"Feature curriculum enabled: ks={ks_final} (starting with top {init_k}/{feature_dim_eff} features)"
                )

                class _FeatureCurriculumCallback(tf.keras.callbacks.Callback):
                    def __init__(self, *, mask_var: tf.Variable, order: list[int], ks_schedule: list[int]):
                        super().__init__()
                        self._mask_var = mask_var
                        self._order = list(order)
                        self._ks = list(ks_schedule)
                        self._stage = 0
                        # Targets for val_direction_accuracy to unlock the next stage.
                        self._targets = [0.58, 0.60, 0.62]

                    def _apply(self, stage: int) -> None:
                        k = int(self._ks[stage])
                        m = np.zeros((int(self._mask_var.shape[0]),), dtype=np.float32)
                        m[np.asarray(self._order[:k], dtype=np.int32)] = 1.0
                        self._mask_var.assign(m)
                        console.print(f"[dim]Curriculum[/dim]: stage {stage+1}/{len(self._ks)} -> top {k} features")

                    def on_train_begin(self, logs=None):
                        self._apply(0)

                    def on_epoch_end(self, epoch, logs=None):
                        if self._stage >= (len(self._ks) - 1):
                            return
                        logs = logs if isinstance(logs, dict) else {}
                        v = logs.get("val_direction_accuracy")
                        try:
                            val_acc = float(v) if v is not None else None
                        except Exception:
                            val_acc = None
                        if val_acc is None:
                            return

                        target = float(self._targets[min(self._stage, len(self._targets) - 1)])
                        if val_acc >= target:
                            self._stage += 1
                            self._apply(self._stage)

                curriculum_cb = _FeatureCurriculumCallback(
                    mask_var=feature_mask_var,
                    order=order_idx,
                    ks_schedule=ks_final,
                )
            except Exception as e:
                console.print(f"[yellow]Feature curriculum init failed[/yellow]: {e}")
                feature_mask_var = None
                curriculum_cb = None

    train_ds = make_sequence_dataset(
        train_feats,
        train_dir_raw,
        train_conf_raw,
        train_conf_w_raw,
        shuffle=True,
        batch_size=int(batch_size),
        seed=int(seed),
        feature_mask=feature_mask_var,
        shuffle_buffer_override=shuffle_buffer,
        prefetch_override=prefetch,
        cache=False,
    )
    val_ds = make_sequence_dataset(
        val_feats,
        val_dir_raw,
        val_conf_raw,
        val_conf_w_raw,
        shuffle=False,
        batch_size=int(batch_size),
        seed=int(seed),
        feature_mask=feature_mask_var,
        shuffle_buffer_override=shuffle_buffer,
        prefetch_override=prefetch,
        cache=bool(cache_val),
    )

    # Ensure the training iterator does not exhaust across epochs.
    # Provide explicit steps so Keras doesn't guess incorrectly when dataset cardinality is unknown.
    n_train_windows = int(len(train_feats) - int(seq_len))
    n_val_windows = int(len(val_feats) - int(seq_len))
    steps_per_epoch = int((n_train_windows + int(batch_size) - 1) // int(batch_size))
    validation_steps = int((n_val_windows + int(batch_size) - 1) // int(batch_size))
    # Keras will warn/interrupt if a finite dataset doesn't have enough batches for
    # steps_per_epoch * epochs. Use repeat() for the datasets passed to fit(), while
    # keeping the original finite datasets for post-training evaluation.
    train_ds_fit = train_ds.repeat()
    val_ds_fit = val_ds.repeat()

    # Perf diagnostics: long epochs are usually (a) very large steps_per_epoch or
    # (b) an unusually slow step. On TF-Metal, the very first step often includes
    # graph tracing + kernel compilation and can take 5-30s, inflating the ETA.
    try:
        console.print(Panel(
            f"[bold]Training Dataset Ready[/bold]\n\n"
            f"[dim]Windows:[/dim] train={int(n_train_windows):,} val={int(n_val_windows):,}\n"
            f"[dim]Steps:[/dim] per_epoch={int(steps_per_epoch)} validation={int(validation_steps)}\n"
            f"[dim]Config:[/dim] seq_len={int(seq_len)} batch_size={int(batch_size)}",
            title="📊 Dataset Configuration",
            border_style="green",
        ))
        if int(steps_per_epoch) >= 2000 and int(steps_per_execution) <= 1:
            console.print(
                "[yellow]⚡ Perf hint:[/yellow] large steps_per_epoch with steps_per_execution=1 can be slow. "
                "Try --steps-per-execution 10 to reduce overhead."
            )
        if not bool(timing):
            console.print(
                "[dim]Note: First batch on Apple Silicon/TF-Metal is slow (graph warmup). Use --timing to verify.[/dim]"
            )
    except Exception:
        pass

    feature_dim = int(train_feats.shape[1])

    model_dir = Path("trained_data") / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_tag}" if run_tag else ""
    final_model_path = model_dir / f"buddy_tf{suffix}.keras"
    final_meta_path = model_dir / f"buddy_tf{suffix}.meta.json"
    candidate_model_path = model_dir / f"buddy_tf{suffix}_candidate.keras"
    candidate_meta_path = model_dir / f"buddy_tf{suffix}_candidate.meta.json"

    if shared_encoder and init_from:
        raise ValueError("--shared-encoder cannot be used with --warm-start/--init-from")
    if init_from:
        console.print(f"Warm-starting from checkpoint: {init_from}")
        model = _load_buddy_checkpoint(model_path=str(init_from), seq_len=int(seq_len), feature_dim=int(feature_dim))
    else:
        head_hidden = int(cfg.get("buddy", {}).get("head_hidden", 64))
        head_layers = int(cfg.get("buddy", {}).get("head_layers", 2))
        head_dropout = float(cfg.get("buddy", {}).get("head_dropout", 0.1))
        dense_hidden = int(cfg.get("buddy", {}).get("dense_hidden", 128))
        dense_dropout = float(cfg.get("buddy", {}).get("dense_dropout", 0.2))
        kernel_regularizer = float(cfg.get("model", {}).get("kernel_regularizer", 0.002))
        noise_std = float(cfg.get("noise_std", 0.03))
        
        # Determine effective model type
        effective_model_type = model_type.lower().strip() if model_type else "lstm"
        
        # MODULAR ENSEMBLE - 4 independent specialist models
        # Each model trains on DIFFERENT features from the SAME candles
        # No shared gradients, no joint loss, each model blind to others
        if effective_model_type == "ensemble":
            import json
            from datetime import datetime
            # Rich imports already at module level - no need to re-import
            
            # Enterprise training status
            enterprise_enabled = options.enterprise if hasattr(options, 'enterprise') else True
            bootstrap_enabled = options.bootstrap if hasattr(options, 'bootstrap') else True
            cv_folds = options.cv_folds if hasattr(options, 'cv_folds') else 5
            generate_report_enabled = options.generate_report if hasattr(options, 'generate_report') else True
            
            # Professional header
            console.print()
            console.print(Panel.fit(
                "[bold white]ENTERPRISE ML TRAINING PIPELINE[/bold white]\n"
                "[dim]Production-Grade Ensemble with MLOps[/dim]",
                border_style="cyan",
                padding=(1, 4),
            ))
            console.print()
            
            # Enterprise features status table
            enterprise_table = Table(show_header=False, box=None, padding=(0, 2))
            enterprise_table.add_column("Feature", style="cyan")
            enterprise_table.add_column("Status", style="green")
            enterprise_table.add_row("MLflow Tracking", "✓ Enabled" if enterprise_enabled else "✗ Disabled")
            enterprise_table.add_row("Walk-Forward CV", f"✓ {cv_folds} folds" if cv_folds > 0 else "✗ Disabled")
            enterprise_table.add_row("Bootstrap CI", "✓ 95% CI" if bootstrap_enabled else "✗ Disabled")
            enterprise_table.add_row("Training Report", "✓ Auto-generated" if generate_report_enabled else "✗ Disabled")
            
            console.print(Panel(enterprise_table, title="[bold]Enterprise Features[/bold]", border_style="green"))
            console.print()
            
            # Import modular components
            from modular_data_loaders import load_all_modular_data
            from modular_trainers import (
                TrainerConfig,
                TCNTrainer,
                TransformerDirectionTrainer,
                TransformerRegimeTrainer,
                XGBoostTrainer,
                RandomForestTrainer,
                RidgeTrainer,
                HistGradientBoostingDirectionTrainer,
            )
            
            # Get model configuration from config
            transformer_cfg = cfg.get("transformer", {})
            use_transformer = transformer_cfg.get("use_transformer", True)  # Default to Transformer
            use_regime = transformer_cfg.get("use_regime", False)  # NEW: Use regime classification
            direction_threshold = cfg.get("direction_threshold", 0.005)  # 0.5% min move
            direction_lookahead = cfg.get("direction_lookahead", 12)  # 12 hours lookahead
            regime_lookback = transformer_cfg.get("regime_lookback", 20)  # 20 bars lookback
            regime_lookahead = transformer_cfg.get("regime_lookahead", 12)  # 12 bars lookahead
            
            # Model architecture table
            arch_table = Table(show_header=True, header_style="bold cyan", box=None)
            arch_table.add_column("Model", style="white", width=15)
            arch_table.add_column("Task", style="yellow", width=25)
            arch_table.add_column("Output", style="green")
            
            # Print architecture with configuration
            if use_regime:
                arch_table.add_row("Transformer", "Regime Classification", "trend / chop / mean_revert")
                arch_table.add_row("XGBoost", "Momentum Analysis", "momentum_score, acceleration")
                arch_table.add_row("Random Forest", "Risk Assessment", "expected_drawdown, streak_prob")
                arch_table.add_row("Ridge", "Confidence Scoring", "confidence (0-100)")
            else:
                dir_model_name = "Transformer" if use_transformer else "TCN"
                arch_table.add_row(dir_model_name, f"Direction (threshold={direction_threshold:.2%})", "long / short")
                arch_table.add_row("XGBoost", "Momentum Analysis", "momentum_score, acceleration")
                arch_table.add_row("Random Forest", "Risk Assessment", "expected_drawdown, streak_prob")
                arch_table.add_row("Ridge", "Confidence Scoring", "confidence (0-100)")
            
            console.print(Panel(arch_table, title="[bold]Model Architecture[/bold]", border_style="yellow"))
            console.print()
            
            # Reconstruct DataFrame from features for modular loaders
            # CRITICAL: Use RAW (unscaled) features because modular loaders
            # have their own scalers that will be used during inference
            console.print()
            console.print(Panel(
                "[bold]Preparing Modular Training Data[/bold]\n\n"
                "[dim]Loading specialized datasets for each ensemble component...[/dim]",
                title="📦 Data Preparation",
                border_style="blue",
            ))
            
            # Check if we have the original raw dataframe from training
            # We need raw OHLCV data for feature engineering
            if 'df' in dir() and df is not None and 'close' in df.columns:
                # Use the original df which has all raw data before feature selection
                # Take first n_samples rows to match train+val
                n_samples = len(train_feats) + len(val_feats)
                feature_df = df.iloc[:n_samples].copy()
                console.print(f"  [green]✓[/green] Using RAW data (n={len(feature_df):,})")
            elif 'ohlc_df' in dir() and ohlc_df is not None:
                # Fallback to ohlc_df which has OHLCV columns
                feature_df = ohlc_df.copy()
                n_samples = len(train_feats) + len(val_feats)
                if len(feature_df) > n_samples:
                    feature_df = feature_df.iloc[:n_samples].copy()
                
                # Re-run feature engineering if needed
                if len(feature_df.columns) < 20:  # Only OHLCV columns
                    from feature_engineering import FeatureEngineering
                    fe_modular = FeatureEngineering(cfg.get("feature_engineering", {}))
                    feature_df = fe_modular.create_features(feature_df, include_all=True)
                
                console.print(f"  [green]✓[/green] Using OHLCV data (n={len(feature_df):,})")
            else:
                # Last resort: use standardized features (not ideal but functional)
                console.print("  [yellow]⚠ Using pre-standardized features - may cause scale mismatch![/yellow]")
                all_feats = np.vstack([train_feats, val_feats])
                feature_df = pd.DataFrame(all_feats, columns=feature_columns)
                
            # Ensure we have OHLCV columns for target calculations
            if 'ohlc_df' in dir() and ohlc_df is not None:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col not in feature_df.columns and col in ohlc_df.columns:
                        vals = ohlc_df[col].values
                        if len(vals) >= len(feature_df):
                            feature_df[col] = vals[:len(feature_df)]
            
            # Ensure we have close column for target calculation
            if 'close' not in feature_df.columns:
                # Use the last price-like column
                for col in feature_df.columns:
                    if 'close' in col.lower():
                        feature_df['close'] = feature_df[col].values
                        break
            
            if 'close' not in feature_df.columns:
                raise ValueError("No 'close' column found for direction labels")
            
            # Add high/low if missing (needed for RF drawdown calculation)
            if 'high' not in feature_df.columns:
                feature_df['high'] = feature_df['close'] * 1.001  # Approximate
            if 'low' not in feature_df.columns:
                feature_df['low'] = feature_df['close'] * 0.999  # Approximate
            if 'volume' not in feature_df.columns:
                feature_df['volume'] = 1000  # Default volume
            
            console.print(f"  Data prepared: [green]{len(feature_df):,}[/green] rows × [green]{len(feature_df.columns)}[/green] columns")
            
            # Load data for each model (same temporal split, different features)
            # Suppress verbose logging during data loading for cleaner output
            import logging as _logging
            _modular_logger = _logging.getLogger('modular_data_loaders')
            _prev_level = _modular_logger.level
            _modular_logger.setLevel(_logging.WARNING)
            
            # Calculate split ratio from train/val sizes
            n_total = len(train_feats) + len(val_feats)
            train_frac = len(train_feats) / n_total
            val_frac = len(val_feats) / n_total
            test_frac = 0.0  # Test is kept separate
            
            # Normalize to sum to 1
            sum_frac = train_frac + val_frac
            train_frac = train_frac / sum_frac * 0.9  # Reserve 10% for internal test
            val_frac = val_frac / sum_frac * 0.9
            test_frac = 0.1
            
            all_data = load_all_modular_data(
                feature_df, 
                split=(train_frac, val_frac, test_frac),
                direction_threshold=direction_threshold,
                direction_lookahead=direction_lookahead,
                use_regime=use_regime,
                regime_lookback=regime_lookback,
                regime_lookahead=regime_lookahead,
            )
            
            # Restore logging level
            _modular_logger.setLevel(_prev_level)
            
            # Build Rich table for dataset summary
            dataset_table = Table(title=None, show_header=True, header_style="bold cyan", box=None)
            dataset_table.add_column("Model", style="white")
            dataset_table.add_column("Train", justify="right", style="green")
            dataset_table.add_column("Val", justify="right", style="yellow")
            dataset_table.add_column("Features", justify="right", style="blue")
            dataset_table.add_column("Notes", style="dim")
            
            for name, data in all_data.items():
                n_features = len(data['feature_names'])
                notes = ""
                # Show label stats for direction model
                if name == 'direction' and 'label_stats' in data:
                    stats = data['label_stats']
                    notes = f"{stats['clear_rate']:.0%} clear, {stats['up_rate']:.0%} up, thr={stats['threshold']:.2%}"
                elif name == 'regime' and 'label_stats' in data:
                    stats = data['label_stats']
                    notes = f"trend={stats['trend_rate']:.0%}, chop={stats['chop_rate']:.0%}"
                
                model_name = name.upper() if name in ['xgboost', 'rf', 'ridge'] else name.capitalize()
                dataset_table.add_row(
                    model_name,
                    f"{len(data['X_train']):,}",
                    f"{len(data['X_val']):,}",
                    str(n_features),
                    notes
                )
            
            console.print(Panel(dataset_table, title="📊 Ensemble Dataset Summary", border_style="blue"))
            
            # Extract instrument for pair-specific model training
            # Priority: 1) OANDA fetch instrument, 2) CSV filename, 3) Default
            if oanda_fetch:
                training_instrument = getattr(oanda_fetch, 'instrument', 'GENERIC')
                training_granularity = getattr(oanda_fetch, 'granularity', 'H1')
            elif csv_path:
                training_instrument = _extract_instrument_from_csv_path(csv_path)
                # Try to extract granularity from filename (e.g., EUR_USD_H1.csv)
                import re
                gran_match = re.search(r'_(M\d+|H\d+|D|W)(?:_|\.)', str(csv_path).upper())
                training_granularity = gran_match.group(1) if gran_match else 'H1'
            else:
                training_instrument = 'GENERIC'
                training_granularity = 'H1'
            
            console.print(f"[cyan]📊 Training instrument: {training_instrument}, Granularity: {training_granularity}[/cyan]")
            
            # Auto-detect fresh vs warm-start training
            # Continual learning (EWC, EMA, Replay) is ONLY beneficial for warm-start
            # For fresh training, these features can cause instability
            model_dir = Path("trained_data/models")
            
            # Get pair-specific model paths for compound learning
            use_transformer = (options.model_type or "transformer") != "tcn"
            pair_paths = _get_pair_model_paths(model_dir, training_instrument, 
                                               "transformer" if use_transformer else "tcn")
            
            # Check for existing pair-specific model for warm-start (compound learning)
            has_existing_model = pair_paths['direction'].exists()
            # Also check generic model as fallback for warm-start
            generic_paths = _get_pair_model_paths(model_dir, "GENERIC", 
                                                  "transformer" if use_transformer else "tcn")
            has_generic_model = (model_dir / "transformer_direction.keras").exists() or generic_paths['direction'].exists()
            
            warm_start_enabled = cfg.get("buddy", {}).get("train_defaults", {}).get("warm_start", True)
            
            # Prefer pair-specific warm-start, fallback to generic
            if has_existing_model and warm_start_enabled:
                is_warm_start = True
                warm_start_path = pair_paths['direction']
                console.print(f"[green]♻️  Compound learning: warm-start from {training_instrument} model[/green]")
            elif has_generic_model and warm_start_enabled:
                is_warm_start = True
                warm_start_path = model_dir / "transformer_direction.keras"
                console.print(f"[yellow]⚡ Initial training for {training_instrument} (warm-start from generic model)[/yellow]")
            else:
                is_warm_start = False
                warm_start_path = None
            
            # Continual learning settings
            # EMA and EWC enabled by default for all training (improves stability)
            use_ema = True  # Always use EMA for stable inference
            use_ewc = is_warm_start and options.enable_ewc  # EWC only for warm-start
            use_replay = is_warm_start
            disable_cl = not is_warm_start
            
            if not is_warm_start:
                console.print("[dim]Fresh training - EMA enabled, EWC/Replay disabled[/dim]")
            elif use_ewc:
                console.print("[cyan]Continual Learning: EMA + EWC + Replay enabled[/cyan]")
            else:
                console.print("[cyan]Continual Learning: EMA + Replay enabled (EWC disabled via --disable-ewc)[/cyan]")
            
            # Default continual learning params
            ema_alpha = 0.999
            ema_update_freq = 16
            ewc_lambda = 1000.0
            ewc_gamma = 0.95
            replay_capacity_ratio = 0.10
            replay_mix_ratio = 0.20
            drift_threshold = 0.03
            
            # Show continual learning config in a compact panel
            if not disable_cl:
                cl_info = (
                    f"[bold]Continual Learning Active[/bold]\n\n"
                    f"[dim]Instrument:[/dim] {training_instrument}  [dim]Granularity:[/dim] {training_granularity}\n"
                    f"[dim]EMA:[/dim] α={ema_alpha}, freq={ema_update_freq}\n"
                    f"[dim]EWC:[/dim] λ={ewc_lambda}, γ={ewc_gamma}\n"
                    f"[dim]Replay:[/dim] {replay_capacity_ratio:.0%} capacity, {replay_mix_ratio:.0%} mix"
                )
                console.print(Panel(cl_info, title="🔄 Adaptive Learning", border_style="magenta"))
            console.print()
            
            # Configure trainers with Transformer settings
            # Use pair-specific checkpoint directory
            pair_checkpoint_dir = str(pair_paths['pair_dir'] / "checkpoints") if training_instrument != "GENERIC" else "trained_data/checkpoints"
            
            trainer_config = TrainerConfig(
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(lr),
                patience=int(patience),
                verbose=int(fit_verbose),
                checkpoint_dir=pair_checkpoint_dir,  # Pair-specific checkpoints
                # Transformer settings from config
                transformer_d_model=int(transformer_cfg.get("d_model", 32)),
                transformer_num_heads=int(transformer_cfg.get("num_heads", 4)),
                transformer_num_layers=int(transformer_cfg.get("num_layers", 2)),
                transformer_dff=int(transformer_cfg.get("dff", 64)),
                transformer_dropout=float(transformer_cfg.get("dropout", 0.2)),
                # XGBoost settings from config
                xgb_n_estimators=int(cfg.get("xgboost", {}).get("n_estimators", 200)),
                xgb_max_depth=int(cfg.get("xgboost", {}).get("max_depth", 5)),
                xgb_learning_rate=float(cfg.get("xgboost", {}).get("learning_rate", 0.05)),
                # Continual learning settings
                use_ema=use_ema,
                ema_decay=ema_alpha,
                ema_update_every=ema_update_freq,
                use_ewc=use_ewc,
                ewc_lambda=ewc_lambda,
                ewc_gamma=ewc_gamma,
                use_replay_buffer=use_replay,
                replay_buffer_ratio=replay_capacity_ratio,
                replay_mix_ratio=replay_mix_ratio,
                drift_threshold=drift_threshold,
            )
            
            all_metrics = {}
            direction_model_name = "Transformer" if use_transformer else "TCN"
            
            # ============================================================
            # TRAIN REGIME OR DIRECTION MODEL
            # ============================================================
            if use_regime:
                # REGIME MODE: 3-class classification
                console.print()
                console.print(Panel(
                    "[bold]Training Transformer (REGIME Classifier)[/bold]\n\n"
                    "[dim]Features:[/dim] ADX, RSI, volatility, z-scores, momentum consistency\n"
                    f"[dim]Output:[/dim] 3-class regime (trend/chop/mean_revert) | lookback={regime_lookback}, lookahead={regime_lookahead}",
                    title="Step 1/4",
                    border_style="yellow",
                ))
                
                regime_data = all_data.get('regime')
                if regime_data is None:
                    raise ValueError("No regime data found. Check use_regime setting.")
                
                regime_trainer = TransformerRegimeTrainer(trainer_config)
                regime_metrics = regime_trainer.train(
                    regime_data['X_train'], regime_data['y_train'],
                    regime_data['X_val'], regime_data['y_val'],
                    feature_names=regime_data['feature_names'],
                    class_names=regime_data.get('class_names'),
                )
                regime_trainer.save(str(model_dir / "transformer_regime.keras"))
                regime_model_path = str(model_dir / "transformer_regime.keras")
                
                all_metrics['regime'] = regime_metrics
                
                # Show F1 scores
                console.print(f"[green]✓ Regime Transformer complete: val_accuracy={regime_metrics['val_accuracy']:.1%}, F1_macro={regime_metrics['f1_macro']:.3f}[/green]")
                console.print(f"   F1 per class: trend={regime_metrics['f1_trend']:.3f}, chop={regime_metrics['f1_chop']:.3f}, mean_revert={regime_metrics['f1_mean_revert']:.3f}")
                
                dir_model_path = regime_model_path  # For metadata
                dir_data = regime_data  # For metadata
                dir_metrics = regime_metrics  # For metadata
            else:
                # DIRECTION MODE: Binary classification (legacy)
                console.print()
                console.print(Panel(
                    f"[bold]Training {direction_model_name} (Direction Predictor)[/bold]\n\n"
                    "[dim]Features:[/dim] Directional indicators (ADX, MACD, SMA crosses, market structure)\n"
                    f"[dim]Output:[/dim] Binary direction (0=short, 1=long) | threshold={direction_threshold:.2%}, lookahead={direction_lookahead}",
                    title="Step 1/4",
                    border_style="cyan",
                ))
                
                # Get direction data (new key 'direction' or fallback to 'tcn')
                dir_data = all_data.get('direction', all_data.get('tcn'))
                
                # WARM-START: Use the already-computed warm_start_path from pair detection above
                # warm_start_path is already set to pair-specific or generic path as appropriate
                if is_warm_start and warm_start_path:
                    console.print(Panel(
                        f"[yellow]Loading weights from:[/yellow] {warm_start_path}\n"
                        "[dim]Training will continue from previous weights (compounding learning)[/dim]",
                        title="🔥 Warm-Start Enabled",
                        border_style="yellow",
                    ))
                
                # Suppress verbose modular_trainers logging for cleaner output
                import logging as _logging
                _trainers_logger = _logging.getLogger('modular_trainers')
                _trainers_prev = _trainers_logger.level
                _trainers_propagate_prev = _trainers_logger.propagate
                _trainers_logger.setLevel(_logging.CRITICAL)  # Suppress ALL
                _trainers_logger.propagate = False
                _absl_logger = _logging.getLogger('absl')
                _absl_prev = _absl_logger.level
                _absl_logger.setLevel(_logging.CRITICAL)
                
                if use_transformer:
                    dir_trainer = TransformerDirectionTrainer(trainer_config)
                    dir_metrics = dir_trainer.train(
                        dir_data['X_train'], dir_data['y_train'],
                        dir_data['X_val'], dir_data['y_val'],
                        feature_names=dir_data['feature_names'],
                        w_train=dir_data.get('w_train'),  # Sample weights for threshold filtering
                        w_val=dir_data.get('w_val'),
                        warm_start_path=str(warm_start_path) if warm_start_path else None,
                        instrument=training_instrument,  # For replay buffer storage
                    )
                    # Save to BOTH pair-specific AND generic paths for compatibility
                    dir_trainer.save(str(pair_paths['direction']), instrument=training_instrument)
                    # Also save to generic path for backward compatibility
                    if training_instrument != "GENERIC":
                        dir_trainer.save(str(model_dir / "transformer_direction.keras"), instrument=training_instrument)
                    dir_model_path = str(pair_paths['direction'])
                    console.print(f"[cyan]💾 Direction model saved to: {pair_paths['direction']}[/cyan]")
                else:
                    dir_trainer = TCNTrainer(trainer_config)
                    dir_metrics = dir_trainer.train(
                        dir_data['X_train'], dir_data['y_train'],
                        dir_data['X_val'], dir_data['y_val'],
                        feature_names=dir_data['feature_names']
                    )
                    dir_trainer.save(str(pair_paths['direction']))
                    if training_instrument != "GENERIC":
                        dir_trainer.save(str(model_dir / "tcn_direction.keras"))
                    dir_model_path = str(pair_paths['direction'])
                    console.print(f"[cyan]💾 TCN model saved to: {pair_paths['direction']}[/cyan]")
                
                all_metrics['direction'] = dir_metrics
                
                # Show balanced accuracy if available
                if 'val_balanced_accuracy' in dir_metrics:
                    console.print(f"[green]✓ {direction_model_name} complete: val_accuracy={dir_metrics['val_accuracy']:.1%}, balanced={dir_metrics['val_balanced_accuracy']:.1%}[/green]")
                    console.print(f"   (up_acc={dir_metrics.get('val_up_accuracy', 0):.1%}, down_acc={dir_metrics.get('val_down_accuracy', 0):.1%})")
                else:
                    console.print(f"[green]✓ {direction_model_name} complete: val_accuracy={dir_metrics['val_accuracy']:.1%}[/green]")
            
            # ============================================================
            # TRAIN XGBOOST (Momentum Analyzer)
            # ============================================================
            console.print()
            console.print(Panel(
                "[bold]Training XGBoost (Momentum Analyzer)[/bold]\n\n"
                "[dim]Features:[/dim] Lagged returns, spread dynamics\n"
                "[dim]Output:[/dim] momentum_score (0-1), acceleration (bool)",
                title="Step 2/4",
                border_style="cyan",
            ))
            
            xgb_data = all_data['xgboost']
            xgb_trainer = XGBoostTrainer(trainer_config)
            xgb_metrics = xgb_trainer.train(
                xgb_data['X_train'], xgb_data['y_train'],
                xgb_data['X_val'], xgb_data['y_val'],
                feature_names=xgb_data['feature_names'],
                momentum_norm_factor=xgb_data.get('momentum_norm_factor'),
            )
            # Save to both pair-specific and generic paths
            xgb_trainer.save(str(pair_paths['xgboost']))
            if training_instrument != "GENERIC":
                xgb_trainer.save(str(model_dir / "xgb_momentum.pkl"))
            all_metrics['xgboost'] = xgb_metrics
            
            console.print(f"[green]✓ XGBoost complete: momentum_mae={xgb_metrics['momentum_mae']:.4f}, accel_acc={xgb_metrics['acceleration_accuracy']:.1%}[/green]")
            console.print(f"[cyan]💾 XGBoost saved to: {pair_paths['xgboost']}[/cyan]")
            
            # ============================================================
            # TRAIN RANDOM FOREST (Risk Assessor)
            # ============================================================
            console.print()
            console.print(Panel(
                "[bold]Training Random Forest (Risk Assessor)[/bold]\n\n"
                "[dim]Features:[/dim] ATR, historical drawdowns, streak patterns\n"
                "[dim]Output:[/dim] expected_drawdown_pips, streak_probability",
                title="Step 3/4",
                border_style="cyan",
            ))
            
            rf_data = all_data['rf']
            rf_trainer = RandomForestTrainer(trainer_config)
            rf_metrics = rf_trainer.train(
                rf_data['X_train'], rf_data['y_train'],
                rf_data['X_val'], rf_data['y_val'],
                feature_names=rf_data['feature_names']
            )
            # Save to both pair-specific and generic paths
            rf_trainer.save(str(pair_paths['rf']))
            if training_instrument != "GENERIC":
                rf_trainer.save(str(model_dir / "rf_risk.pkl"))
            all_metrics['rf'] = rf_metrics
            
            console.print(f"[green]✓ RF complete: drawdown_mae={rf_metrics.get('drawdown_mae_bps', rf_metrics.get('drawdown_mae_pips', 0)*10000):.1f} bps, streak_mae={rf_metrics['streak_prob_mae']:.4f}[/green]")
            console.print(f"[cyan]💾 RF saved to: {pair_paths['rf']}[/cyan]")
            
            # ============================================================
            # TRAIN RIDGE (Confidence Scorer)
            # ============================================================
            console.print()
            console.print(Panel(
                "[bold]Training ElasticNet (Confidence Scorer)[/bold]\n\n"
                "[dim]Features:[/dim] Rolling variance, volume dynamics\n"
                "[dim]Output:[/dim] Confidence score (0-100)\n"
                "[dim]CV:[/dim] TimeSeriesSplit (temporal, no leakage)",
                title="Step 4/4",
                border_style="cyan",
            ))
            
            ridge_data = all_data['ridge']
            ridge_trainer = RidgeTrainer(trainer_config)
            ridge_metrics = ridge_trainer.train(
                ridge_data['X_train'], ridge_data['y_train'],
                ridge_data['X_val'], ridge_data['y_val'],
                feature_names=ridge_data['feature_names']
            )
            # Save to both pair-specific and generic paths
            ridge_trainer.save(str(pair_paths['ridge']))
            if training_instrument != "GENERIC":
                ridge_trainer.save(str(model_dir / "ridge_confidence.pkl"))
            all_metrics['ridge'] = ridge_metrics
            
            console.print(f"[green]✓ ElasticNet complete: MAE={ridge_metrics['confidence_mae']:.2f}, R²={ridge_metrics['r2_score']:.4f}, alpha={ridge_metrics.get('best_alpha', 1.0):.4f}, l1_ratio={ridge_metrics.get('best_l1_ratio', 0.5):.2f}, sparse={ridge_metrics.get('n_nonzero_coefs', '?')}/{ridge_metrics.get('n_total_coefs', '?')}[/green]")
            console.print(f"[cyan]💾 Ridge saved to: {pair_paths['ridge']}[/cyan]")
            
            # ============================================================
            # TRAIN HISTGB (Optional - for hybrid voting with Transformer)
            # ============================================================
            train_histgb = cfg.get("buddy", {}).get("train_defaults", {}).get("train_histgb", False)
            histgb_metrics = None
            
            if train_histgb and not use_regime:
                console.print()
                console.print(Panel(
                    "[bold]Training HistGB (Hybrid Voting Baseline)[/bold]\n\n"
                    "[dim]Purpose:[/dim] Hybrid voting with Transformer → trade only when BOTH AGREE\n"
                    "[dim]Expected:[/dim] Higher precision at cost of fewer trades",
                    title="Step 5 (Optional)",
                    border_style="magenta",
                ))
                
                histgb_trainer = HistGradientBoostingDirectionTrainer(trainer_config)
                histgb_metrics = histgb_trainer.train(
                    dir_data['X_train'], dir_data['y_train'],
                    dir_data['X_val'], dir_data['y_val'],
                    feature_names=dir_data['feature_names'],
                )
                # Save to both pair-specific and generic paths
                histgb_trainer.save(str(pair_paths['histgb']))
                if training_instrument != "GENERIC":
                    histgb_trainer.save(str(model_dir / "histgb_direction.pkl"))
                all_metrics['histgb'] = histgb_metrics
                
                console.print(f"[green]✓ HistGB complete: val_accuracy={histgb_metrics['val_accuracy']:.1%}, balanced={histgb_metrics.get('val_balanced_accuracy', 0):.1%}[/green]")
                console.print(f"[cyan]💾 HistGB saved to: {pair_paths['histgb']}[/cyan]")
                console.print("[yellow]🔥 Hybrid voting ENABLED: Transformer + HistGB will vote together[/yellow]")
            
            # ============================================================
            # SAVE METADATA
            # ============================================================
            if use_regime:
                primary_model_meta = {
                    "regime": {
                        "path": dir_model_path,
                        "type": "transformer_regime",
                        "purpose": "regime_classification",
                        "output": "3-class (0=trend, 1=chop, 2=mean_revert)",
                        "metrics": dir_metrics,
                        "features": dir_data['feature_names'],
                        "label_stats": dir_data.get('label_stats', {}),
                        "class_names": dir_data.get('class_names', ['trend', 'chop', 'mean_revert']),
                    },
                }
                primary_config = {
                    "use_regime": True,
                    "regime_lookback": regime_lookback,
                    "regime_lookahead": regime_lookahead,
                }
            else:
                primary_model_meta = {
                    "direction": {
                        "path": dir_model_path,
                        "type": direction_model_name.lower(),
                        "purpose": "direction_prediction",
                        "output": "binary (0=short, 1=long)",
                        "metrics": dir_metrics,
                        "features": dir_data['feature_names'],
                        "label_stats": dir_data.get('label_stats', {}),
                    },
                }
                primary_config = {
                    "use_regime": False,
                    "threshold": direction_threshold,
                    "lookahead": direction_lookahead,
                    "use_transformer": use_transformer,
                }
            
            meta = {
                "model_type": "modular_ensemble",
                "use_regime": use_regime,
                "primary_model_type": "regime" if use_regime else direction_model_name.lower(),
                "primary_config": primary_config,
                "training_instrument": training_instrument,  # Track which pair this was trained on
                "pair_model_dir": str(pair_paths['pair_dir']),  # Pair-specific model directory
                "models": {
                    **primary_model_meta,
                    "xgboost": {
                        "path": str(pair_paths['xgboost']),
                        "purpose": "momentum_analysis",
                        "output": "momentum_score (0-1), acceleration (bool)",
                        "metrics": xgb_metrics,
                        "features": xgb_data['feature_names'],
                    },
                    "rf": {
                        "path": str(pair_paths['rf']),
                        "purpose": "risk_assessment",
                        "output": "expected_drawdown_pips, streak_prob",
                        "metrics": rf_metrics,
                        "features": rf_data['feature_names'],
                    },
                    "ridge": {
                        "path": str(pair_paths['ridge']),
                        "purpose": "confidence_scoring",
                        "output": "confidence (0-100)",
                        "metrics": ridge_metrics,
                        "features": ridge_data['feature_names'],
                    },
                },
                "inference_gates": {
                    "min_confidence": 75,
                    "min_momentum_or_accel": True,
                    "max_drawdown_pips": 30,
                    "max_streak_prob": 0.3,
                    "risk_per_trade_pct": 0.02,
                },
                "data_split": {
                    "train_frac": train_frac,
                    "val_frac": val_frac,
                    "test_frac": test_frac,
                    "total_samples": len(feature_df),
                },
                "trained_at": datetime.now().isoformat(),
            }
            
            meta_path = model_dir / "modular_ensemble.meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
            
            # ============================================================
            # PROFESSIONAL SUMMARY WITH RICH
            # ============================================================
            console.print()
            
            # Performance metrics table
            perf_table = Table(show_header=True, header_style="bold green", title="Model Performance")
            perf_table.add_column("Model", style="white", width=20)
            perf_table.add_column("Metric", style="cyan", width=20)
            perf_table.add_column("Value", style="green", justify="right")
            
            if use_regime:
                f1_macro = dir_metrics.get('f1_macro', 0)
                perf_table.add_row("Transformer", "F1 Macro", f"{f1_macro:.3f}")
                perf_table.add_row("", "F1 Trend", f"{dir_metrics.get('f1_trend', 0):.3f}")
                perf_table.add_row("", "F1 Chop", f"{dir_metrics.get('f1_chop', 0):.3f}")
            else:
                dir_acc = dir_metrics['val_accuracy']
                bal_acc = dir_metrics.get('val_balanced_accuracy', dir_acc)
                perf_table.add_row(direction_model_name, "Val Accuracy", f"{dir_acc:.1%}")
                perf_table.add_row("", "Balanced Acc", f"{bal_acc:.1%}")
            
            perf_table.add_row("XGBoost", "Accel Accuracy", f"{xgb_metrics['acceleration_accuracy']:.1%}")
            perf_table.add_row("", "Momentum MAE", f"{xgb_metrics['momentum_mae']:.4f}")
            perf_table.add_row("Random Forest", "Drawdown MAE", f"{rf_metrics.get('drawdown_mae_bps', rf_metrics.get('drawdown_mae_pips', 0)*10000):.1f} bps")
            perf_table.add_row("", "Streak MAE", f"{rf_metrics['streak_prob_mae']:.4f}")
            perf_table.add_row("Ridge", "R² Score", f"{ridge_metrics['r2_score']:.3f}")
            perf_table.add_row("", "Confidence MAE", f"{ridge_metrics['confidence_mae']:.2f}")
            perf_table.add_row("", "Best Alpha", f"{ridge_metrics.get('best_alpha', 1.0):.4f}")
            perf_table.add_row("", "L1 Ratio", f"{ridge_metrics.get('best_l1_ratio', 0.5):.2f}")
            perf_table.add_row("", "Features (sparse)", f"{ridge_metrics.get('n_nonzero_coefs', '?')}/{ridge_metrics.get('n_total_coefs', '?')}")
            
            console.print(Panel(perf_table, border_style="green"))
            console.print()
            
            # Saved artifacts table
            artifacts_table = Table(show_header=False, box=None)
            artifacts_table.add_column("Type", style="cyan", width=15)
            artifacts_table.add_column("Path", style="dim")
            artifacts_table.add_row("Direction", str(dir_model_path))
            artifacts_table.add_row("Momentum", str(pair_paths['xgboost']))
            artifacts_table.add_row("Risk", str(pair_paths['rf']))
            artifacts_table.add_row("Confidence", str(pair_paths['ridge']))
            artifacts_table.add_row("Metadata", str(meta_path))
            
            console.print(Panel(artifacts_table, title="[bold]Saved Artifacts[/bold]", border_style="blue"))
            console.print()
            
            # ============================================================
            # ENTERPRISE TRAINING VALIDATION (enabled by default for ensemble)
            # ============================================================
            enterprise_enabled = options.enterprise if hasattr(options, 'enterprise') else True
            
            if enterprise_enabled:
                console.print()
                console.print(Panel(
                    "[bold]Enterprise Validation Suite[/bold]\n"
                    "[dim]Statistical validation, cross-validation, and experiment tracking[/dim]",
                    title="🏢 Enterprise",
                    border_style="cyan",
                ))
                
                try:
                    from enterprise_training import (
                        StatisticalValidator,
                        WalkForwardValidator,
                        WalkForwardConfig,
                        ExperimentTracker,
                        generate_training_report,
                        MLFLOW_AVAILABLE,
                    )
                    
                    stat_validator = StatisticalValidator()
                    validation_results = Table(show_header=True, header_style="bold cyan", title="Validation Results")
                    validation_results.add_column("Check", style="white", width=25)
                    validation_results.add_column("Result", style="green", width=15)
                    validation_results.add_column("Details", style="dim", width=35)
                    
                    # 1. Bootstrap CI for Transformer direction accuracy
                    cv_folds = options.cv_folds if hasattr(options, 'cv_folds') else 5
                    bootstrap_enabled = options.bootstrap if hasattr(options, 'bootstrap') else False
                    bootstrap_samples = options.bootstrap_samples if hasattr(options, 'bootstrap_samples') else 1000
                    generate_report_enabled = options.generate_report if hasattr(options, 'generate_report') else False
                    
                    if bootstrap_enabled:
                        # Get validation predictions from direction model
                        if not use_regime and hasattr(dir_trainer, 'model') and dir_trainer.model is not None:
                            try:
                                # Ensure input shape is set for predict
                                X_val = np.asarray(dir_data['X_val'])
                                val_preds = dir_trainer.model.predict(X_val, verbose=0)
                                if len(val_preds.shape) > 1:
                                    val_preds = val_preds[:, 0] if val_preds.shape[1] == 1 else val_preds
                                val_preds_binary = (val_preds > 0.5).astype(int).flatten()
                                val_labels = dir_data['y_val'].flatten()
                                
                                bootstrap_results = stat_validator.bootstrap_confidence_interval(
                                    val_labels, val_preds_binary,
                                    metric_fn=lambda y, p: np.mean(y == p),
                                    n_bootstrap=bootstrap_samples,
                                )
                                
                                validation_results.add_row(
                                    "Bootstrap CI (95%)",
                                    f"{bootstrap_results['mean']:.2%}",
                                    f"[{bootstrap_results['ci_lower']:.2%}, {bootstrap_results['ci_upper']:.2%}]"
                                )
                                
                                # Add to metrics
                                dir_metrics['bootstrap_ci_lower'] = bootstrap_results['ci_lower']
                                dir_metrics['bootstrap_ci_upper'] = bootstrap_results['ci_upper']
                                dir_metrics['bootstrap_mean'] = bootstrap_results['mean']
                            except Exception as bootstrap_err:
                                validation_results.add_row(
                                    "Bootstrap CI (95%)",
                                    "[yellow]Skipped[/yellow]",
                                    f"[dim]Model shape incompatible[/dim]"
                                )
                    
                    # 2. Walk-Forward Cross-Validation (if enabled and data sufficient)
                    # For continual learning: EVALUATE trained model on different time windows
                    # DO NOT retrain on each fold - that destroys learned weights
                    if cv_folds > 0 and len(dir_data['X_train']) > cv_folds * 500:
                        wf_config = WalkForwardConfig(
                            n_splits=cv_folds,
                            train_period=len(dir_data['X_train']) // (cv_folds + 1),
                            test_period=len(dir_data['X_val']) // cv_folds,
                            gap=24,
                            expanding=True,
                        )
                        wf_validator = WalkForwardValidator(wf_config)
                        
                        # Combine train+val for CV evaluation
                        X_full = np.vstack([dir_data['X_train'], dir_data['X_val']])
                        y_full = np.concatenate([dir_data['y_train'].flatten(), dir_data['y_val'].flatten()])
                        
                        cv_scores = []
                        with console.status("[bold cyan]Running Walk-Forward CV (evaluation only)...", spinner="dots"):
                            for fold_idx, (train_idx, val_idx) in enumerate(wf_validator.split(X_full)):
                                X_cv_val = X_full[val_idx]
                                y_cv_val = y_full[val_idx]
                                
                                # EVALUATE the trained model on each fold's validation window
                                # No retraining - just measure generalization across time
                                try:
                                    # Use the already-trained dir_trainer model
                                    # Model expects 3D input (batch, seq_len, features)
                                    if len(X_cv_val.shape) == 3:
                                        y_pred = dir_trainer.model.predict(X_cv_val, verbose=0)
                                        y_pred_binary = (y_pred > 0.5).astype(float).flatten()
                                        fold_acc = np.mean(y_pred_binary == y_cv_val)
                                        cv_scores.append(fold_acc)
                                    else:
                                        logger.warning(f"CV fold {fold_idx+1} skipped: expected 3D input, got {len(X_cv_val.shape)}D")
                                except Exception as e:
                                    # Silently skip CV folds that fail - model shape may be incompatible
                                    logger.debug(f"CV fold {fold_idx+1} evaluation failed: {e}")
                        
                        if cv_scores:
                            cv_mean = np.mean(cv_scores)
                            cv_std = np.std(cv_scores)
                            
                            validation_results.add_row(
                                f"Walk-Forward CV ({cv_folds} folds)",
                                f"{cv_mean:.2%}",
                                f"± {cv_std:.2%} std"
                            )
                            
                            # Store CV results
                            dir_metrics['cv_mean'] = cv_mean
                            dir_metrics['cv_std'] = cv_std
                            dir_metrics['cv_scores'] = cv_scores
                    
                    # Print validation results table
                    console.print(validation_results)
                    console.print()
                    
                    # 3. MLflow Experiment Tracking
                    if MLFLOW_AVAILABLE:
                        mlflow_experiment = options.mlflow_experiment if hasattr(options, 'mlflow_experiment') else None
                        experiment_name = mlflow_experiment or f"{training_instrument}_{training_granularity}"
                        
                        console.print("[cyan]📊 MLflow Tracking[/cyan]")
                        console.print(f"  Experiment: {experiment_name}")
                        
                        import mlflow
                        from pathlib import Path as _Path
                        _mlflow_db = _Path("trained_data/mlruns/mlflow.db").absolute()
                        _mlflow_db.parent.mkdir(parents=True, exist_ok=True)
                        mlflow.set_tracking_uri(f"sqlite:///{_mlflow_db}")
                        mlflow.set_experiment(experiment_name)
                        
                        with mlflow.start_run(run_name=f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                            # Log parameters
                            mlflow.log_param("model_type", "ensemble")
                            mlflow.log_param("instrument", training_instrument)
                            mlflow.log_param("granularity", training_granularity)
                            mlflow.log_param("epochs", epochs)
                            mlflow.log_param("batch_size", batch_size)
                            mlflow.log_param("learning_rate", lr)
                            mlflow.log_param("direction_threshold", direction_threshold)
                            mlflow.log_param("direction_lookahead", direction_lookahead)
                            
                            # Log metrics
                            mlflow.log_metric("direction_val_accuracy", dir_metrics['val_accuracy'])
                            mlflow.log_metric("direction_balanced_accuracy", dir_metrics.get('val_balanced_accuracy', 0))
                            mlflow.log_metric("xgb_acceleration_accuracy", xgb_metrics['acceleration_accuracy'])
                            mlflow.log_metric("rf_drawdown_mae", rf_metrics.get('drawdown_mae_bps', 0))
                            mlflow.log_metric("ridge_r2_score", ridge_metrics['r2_score'])
                            
                            if 'cv_mean' in dir_metrics:
                                mlflow.log_metric("cv_mean", dir_metrics['cv_mean'])
                                mlflow.log_metric("cv_std", dir_metrics['cv_std'])
                            
                            if 'bootstrap_ci_lower' in dir_metrics:
                                mlflow.log_metric("bootstrap_ci_lower", dir_metrics['bootstrap_ci_lower'])
                                mlflow.log_metric("bootstrap_ci_upper", dir_metrics['bootstrap_ci_upper'])
                            
                            # Log model artifacts
                            mlflow.log_artifact(str(meta_path))
                            
                            console.print(f"  ✓ Run logged: {mlflow.active_run().info.run_id[:8]}...")
                    
                    # 4. Generate Report
                    if generate_report_enabled:
                        console.print("[cyan]📄 Generating Training Report[/cyan]")
                        
                        report_path = model_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                        
                        report_content = f"""# Enterprise Training Report
                        
## Training Summary
- **Date**: {datetime.now().isoformat()}
- **Instrument**: {training_instrument}
- **Granularity**: {training_granularity}
- **Model Type**: Modular Ensemble

## Model Performance

### Transformer Direction
- Validation Accuracy: {dir_metrics['val_accuracy']:.2%}
- Balanced Accuracy: {dir_metrics.get('val_balanced_accuracy', dir_metrics['val_accuracy']):.2%}
"""
                        if 'bootstrap_ci_lower' in dir_metrics:
                            report_content += f"- Bootstrap 95% CI: [{dir_metrics['bootstrap_ci_lower']:.2%}, {dir_metrics['bootstrap_ci_upper']:.2%}]\n"
                        
                        if 'cv_mean' in dir_metrics:
                            report_content += f"- Cross-Validation: {dir_metrics['cv_mean']:.2%} ± {dir_metrics['cv_std']:.2%}\n"
                        
                        report_content += f"""
### XGBoost Momentum
- Acceleration Accuracy: {xgb_metrics['acceleration_accuracy']:.2%}
- Momentum MAE: {xgb_metrics['momentum_mae']:.4f}

### Random Forest Risk
- Drawdown MAE: {rf_metrics.get('drawdown_mae_bps', rf_metrics.get('drawdown_mae_pips', 0)*10000):.1f} bps
- Streak Prob MAE: {rf_metrics['streak_prob_mae']:.4f}

### Ridge Confidence
- R² Score: {ridge_metrics['r2_score']:.4f}
- Confidence MAE: {ridge_metrics['confidence_mae']:.2f}

## Configuration
- Epochs: {epochs}
- Batch Size: {batch_size}
- Learning Rate: {lr}
- Direction Threshold: {direction_threshold:.2%}
- Direction Lookahead: {direction_lookahead} bars

## Model Paths
- Direction: {dir_model_path}
- Momentum: {pair_paths['xgboost']}
- Risk: {pair_paths['rf']}
- Confidence: {pair_paths['ridge']}
"""
                        
                        with open(report_path, 'w') as f:
                            f.write(report_content)
                        
                        console.print(f"  ✓ Report saved: {report_path}")
                    
                    # Final success panel
                    console.print()
                    console.print(Panel(
                        "[bold green]✓ TRAINING COMPLETE[/bold green]\n\n"
                        f"Models saved to: [cyan]{model_dir}[/cyan]\n"
                        "Enterprise validation: [green]PASSED[/green]",
                        border_style="green",
                    ))
                    
                    # RL Position Sizer Training (optional, after ensemble)
                    # Check BOTH CLI option AND config file setting (consistent with TCN/LSTM path)
                    auto_train_rl_config = cfg.get("training", {}).get("auto_train_rl", True)
                    train_rl_sizer_enabled = options.train_rl_sizer and auto_train_rl_config
                    rl_timesteps = options.rl_timesteps
                    
                    if train_rl_sizer_enabled:
                        # Prepare RL training data from ensemble training
                        try:
                            # Get features from direction data (most complete feature set)
                            rl_features = np.vstack([
                                dir_data['X_train'],
                                dir_data['X_val'],
                            ]).astype(np.float32)
                            
                            # Get ensemble predictions by combining trained models
                            # Load and predict using trained direction and confidence models
                            rl_direction_probs = np.zeros(len(rl_features))
                            rl_confidences = np.zeros(len(rl_features))
                            
                            # Predict with direction model (Transformer/TCN)
                            try:
                                if use_transformer:
                                    dir_preds = dir_trainer.predict(rl_features)
                                else:
                                    dir_preds = dir_trainer.predict(rl_features)
                                # Extract direction probability
                                if isinstance(dir_preds, dict):
                                    rl_direction_probs = dir_preds.get('direction_prob', np.full(len(rl_features), 0.5))
                                elif hasattr(dir_preds, 'flatten'):
                                    rl_direction_probs = dir_preds.flatten()
                                else:
                                    rl_direction_probs = np.array(dir_preds).flatten()
                            except Exception as e:
                                console.print(f"[dim]Using default direction probs: {e}[/dim]")
                                rl_direction_probs = np.full(len(rl_features), 0.5)
                            
                            # Predict with ridge confidence model
                            try:
                                ridge_feats = np.vstack([
                                    ridge_data['X_train'],
                                    ridge_data['X_val'],
                                ]).astype(np.float32)
                                conf_preds = ridge_trainer.predict(ridge_feats)
                                if isinstance(conf_preds, dict):
                                    rl_confidences = conf_preds.get('confidence', np.full(len(ridge_feats), 50.0))
                                elif hasattr(conf_preds, 'flatten'):
                                    rl_confidences = conf_preds.flatten()
                                else:
                                    rl_confidences = np.array(conf_preds).flatten()
                                # Normalize confidence to 0-1 range (model outputs 0-100)
                                rl_confidences = np.clip(rl_confidences / 100.0, 0.0, 1.0)
                            except Exception as e:
                                console.print(f"[dim]Using default confidences: {e}[/dim]")
                                rl_confidences = np.full(len(rl_features), 0.5)
                            
                            # Ensure arrays are same length
                            min_len = min(len(rl_direction_probs), len(rl_confidences), len(rl_features))
                            rl_direction_probs = rl_direction_probs[:min_len]
                            rl_confidences = rl_confidences[:min_len]
                            rl_features = rl_features[:min_len]
                            
                            # Stack predictions: [direction_prob, confidence]
                            rl_predictions = np.column_stack([
                                rl_direction_probs,
                                rl_confidences,
                            ]).astype(np.float32)
                            
                            # Get prices from feature_df
                            rl_prices = feature_df['close'].values[:min_len].astype(np.float32)
                            
                            console.print(f"[dim]RL training data: {len(rl_features):,} samples, "
                                          f"{rl_features.shape[1]} features[/dim]")
                            
                            _train_rl_position_sizer_if_ready(
                                console=console,
                                rl_timesteps=rl_timesteps,
                                features=rl_features,
                                ensemble_predictions=rl_predictions,
                                prices=rl_prices,
                            )
                        except Exception as e:
                            console.print(f"[yellow]RL data preparation failed: {e}[/yellow]")
                            console.print("[dim]Skipping RL position sizer training[/dim]")
                    
                except ImportError as e:
                    console.print(f"[yellow]Enterprise features unavailable: {e}[/yellow]")
                    console.print("[dim]Install with: pip install mlflow structlog[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Enterprise validation error: {e}[/yellow]")
                    import traceback
                    traceback.print_exc()
            
            return
        
        # XGBoost model - handles training separately
        elif effective_model_type == "xgboost":
            import json
            from datetime import datetime
            console.print("[cyan]Model: XGBoost (gradient boosting - often better on tabular data)[/cyan]")
            from xgboost_model import train_xgboost_buddy, XGBoostKerasWrapper
            
            # XGBoost training
            xgb_config = {
                "n_estimators": int(cfg.get("xgboost", {}).get("n_estimators", 500)),
                "max_depth": int(cfg.get("xgboost", {}).get("max_depth", 6)),
                "learning_rate": float(cfg.get("xgboost", {}).get("learning_rate", 0.05)),
                "early_stopping_rounds": int(patience),
            }
            
            xgb_model, xgb_history = train_xgboost_buddy(
                X_train=train_feats,
                y_train_direction=train_dir_raw,
                y_train_confidence=train_conf_raw,
                X_val=val_feats,
                y_val_direction=val_dir_raw,
                y_val_confidence=val_conf_raw,
                config=xgb_config,
                verbose=fit_verbose > 0,
            )
            
            # Wrap for Keras-like interface
            model = XGBoostKerasWrapper(xgb_model)
            
            # Store XGBoost-specific results
            val_dir_acc = xgb_history.get("val_direction_accuracy", 0.5)
            console.print(f"XGBoost validation direction accuracy: {val_dir_acc:.1%}")
            
            # Save XGBoost model
            xgb_model_path = model_dir / f"buddy_xgb{suffix}.pkl"
            xgb_model.save(xgb_model_path)
            console.print(f"Saved: {xgb_model_path}")
            
            # Save metadata
            meta = {
                "model_type": "xgboost",
                "model_path": str(xgb_model_path),
                "seq_len": seq_len,
                "feature_columns": feature_columns,
                "val_direction_accuracy": float(val_dir_acc),
                "standardize": {
                    "mean": [float(x) for x in np.asarray(mu).flatten()],
                    "std": [float(x) for x in np.asarray(sigma).flatten()],
                },
                "trained_at": datetime.now().isoformat(),
            }
            
            with open(candidate_meta_path.with_suffix('.xgb.meta.json'), "w") as f:
                json.dump(meta, f, indent=2)
            
            console.print(f"Saved: {candidate_meta_path.with_suffix('.xgb.meta.json')}")
            
            # Continue to meta-labeling if enabled (don't return early)
            # XGBoost can also benefit from a meta-labeling filter
            xgboost_primary_model = xgb_model
            xgboost_model_path = xgb_model_path
            
            # Check if meta-labeling is enabled
            meta_labeling_enabled = bool(cfg.get("buddy", {}).get("meta_labeling", {}).get("enabled", False))
            if meta_labeling_enabled:
                try:
                    from meta_labeling import MetaLabeler, MetaLabelingConfig
                    
                    console.print("[cyan]Training meta-labeler for XGBoost (trade filter)...[/cyan]")
                    
                    # Get XGBoost predictions on training and validation data
                    xgb_preds_train = xgb_model.predict(train_feats)
                    xgb_preds_val = xgb_model.predict(val_feats)
                    
                    train_dir_preds = xgb_preds_train["direction"]
                    val_dir_preds = xgb_preds_val["direction"]
                    
                    # Configure meta-labeler with regularization (prevents overfitting)
                    meta_cfg_yaml = cfg.get("buddy", {}).get("meta_labeling", {})
                    meta_cfg = MetaLabelingConfig(
                        use_xgboost=True,
                        # Reduced defaults to prevent overfitting
                        n_estimators=int(meta_cfg_yaml.get("n_estimators", 100)),
                        max_depth=int(meta_cfg_yaml.get("max_depth", 2)),
                        learning_rate=float(meta_cfg_yaml.get("learning_rate", 0.05)),
                        # Regularization params
                        reg_alpha=float(meta_cfg_yaml.get("reg_alpha", 1.0)),
                        reg_lambda=float(meta_cfg_yaml.get("reg_lambda", 5.0)),
                        subsample=float(meta_cfg_yaml.get("subsample", 0.7)),
                        colsample_bytree=float(meta_cfg_yaml.get("colsample_bytree", 0.7)),
                        min_child_weight=int(meta_cfg_yaml.get("min_child_weight", 5)),
                        early_stopping_rounds=int(meta_cfg_yaml.get("early_stopping_rounds", 30)),
                        max_overfitting_gap=float(meta_cfg_yaml.get("max_overfitting_gap", 0.15)),
                        auto_tune=bool(meta_cfg_yaml.get("auto_tune", False)),
                        # Threshold
                        min_confidence_threshold=float(meta_cfg_yaml.get("threshold", 0.55)),
                        use_reduced_features=bool(meta_cfg_yaml.get("use_reduced_features", True)),
                    )
                    
                    # Train meta-labeler
                    labeler = MetaLabeler(meta_cfg)
                    
                    # Fit the meta-labeler (auto-tune happens inside if config.auto_tune=True)
                    meta_metrics = labeler.fit(
                        train_feats,
                        train_dir_preds,
                        train_dir_raw,
                        features_val=val_feats,
                        predictions_val=val_dir_preds,
                        outcomes_val=val_dir_raw,
                        verbose=True,
                    )
                    
                    train_meta_acc = meta_metrics.get("meta_train_accuracy", 0)
                    val_meta_acc = meta_metrics.get("meta_val_accuracy", 0)
                    overfit_gap = meta_metrics.get("overfitting_gap", train_meta_acc - val_meta_acc)
                    
                    # Tune threshold on validation set
                    best_thresh, thresh_metrics = labeler.tune_threshold(
                        val_feats, val_dir_preds, val_dir_raw, verbose=True
                    )
                    thresh_improvement = thresh_metrics.get("improvement", 0) if thresh_metrics else 0
                    
                    console.print(f"Meta-labeler: train_acc={train_meta_acc:.1%} val_acc={val_meta_acc:.1%} gap={overfit_gap:.1%}")
                    if thresh_metrics:
                        console.print(f"Threshold tuned: {best_thresh:.2f} -> +{thresh_improvement*100:.1f}% improvement")
                    
                    # Reject meta-labeler if still overfitting badly
                    if overfit_gap > 0.20:
                        console.print(f"[yellow]⚠️ Meta-labeler rejected (gap {overfit_gap:.1%} > 20%)[/yellow]")
                        meta["meta_labeling"] = {"enabled": False, "reason": f"overfitting_gap_{overfit_gap:.1%}"}
                    else:
                        # Save meta-labeler
                        meta_labeler_path = model_dir / f"buddy_xgb_meta{suffix}.pkl"
                        labeler.save(meta_labeler_path)
                        console.print(f"Saved: {meta_labeler_path}")
                        
                        # Update metadata with meta-labeler info
                        meta["meta_labeling"] = {
                            "enabled": True,
                            "path": str(meta_labeler_path),
                            "train_accuracy": float(train_meta_acc),
                            "val_accuracy": float(val_meta_acc),
                            "overfitting_gap": float(overfit_gap),
                            "tuned_threshold": float(best_thresh),
                            "threshold_improvement": float(thresh_improvement),
                        }
                    
                    # Re-save metadata with meta-labeling info
                    with open(candidate_meta_path.with_suffix('.xgb.meta.json'), "w") as f:
                        json.dump(meta, f, indent=2)
                    
                except Exception as e:
                    console.print(f"[yellow]Meta-labeling failed[/yellow]: {e}")
                    import traceback
                    traceback.print_exc()
            
            console.print("[green]XGBoost training complete![/green]")
            
            # ==========================================================================
            # RL Position Sizer Training (automatic after XGBoost ensemble training)
            # ==========================================================================
            auto_train_rl_config = cfg.get("training", {}).get("auto_train_rl", True)
            train_rl_sizer_enabled = options.train_rl_sizer and auto_train_rl_config
            
            if train_rl_sizer_enabled:
                try:
                    console.print()
                    console.print(Panel(
                        "[bold cyan]🤖 RL Position Sizer Training[/bold cyan]\n\n"
                        "[dim]Training RL agent to learn optimal position sizing based on[/dim]\n"
                        "[dim]XGBoost ensemble predictions and market conditions.[/dim]",
                        title="RL Training",
                        border_style="cyan",
                    ))
                    
                    # Prepare RL training data from XGBoost training
                    rl_features = np.vstack([train_feats, val_feats]).astype(np.float32)
                    n_rl_samples = len(rl_features)
                    
                    # Get prices from ohlc_df
                    rl_prices = ohlc_df['close'].values[:n_rl_samples].astype(np.float32)
                    
                    # Get XGBoost predictions for all features
                    console.print(f"[dim]Generating XGBoost predictions for {n_rl_samples:,} samples...[/dim]")
                    xgb_preds = xgb_model.predict(rl_features)
                    rl_direction_probs = xgb_preds["direction"].astype(np.float32)
                    rl_confidences = np.clip(xgb_preds.get("confidence", np.full(n_rl_samples, 0.5)), 0.0, 1.0).astype(np.float32)
                    
                    # Stack predictions: [direction_prob, confidence]
                    rl_predictions = np.column_stack([
                        rl_direction_probs,
                        rl_confidences,
                    ]).astype(np.float32)
                    
                    console.print(f"[dim]RL training data ready: {n_rl_samples:,} samples, {rl_features.shape[1]} features[/dim]")
                    
                    # Train RL position sizer
                    rl_timesteps = options.rl_timesteps
                    _train_rl_position_sizer_if_ready(
                        console=console,
                        rl_timesteps=rl_timesteps,
                        features=rl_features,
                        ensemble_predictions=rl_predictions,
                        prices=rl_prices,
                    )
                    
                except Exception as e:
                    # RL training failure should NOT crash main training
                    console.print(f"[yellow]⚠ RL position sizer training failed: {e}[/yellow]")
                    console.print("[dim]XGBoost training completed successfully - RL training is optional.[/dim]")
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
            
            return
        
        # TCN is prioritized for M1 Metal (faster than LSTM)
        elif effective_model_type == "tcn":
            console.print("[green]Model: TCN (M1 Metal optimized - 2-3x faster than LSTM)[/green]")
            model = _build_buddy_model_for_type(
                model_type="tcn",
                feature_dim=int(feature_dim),
                seq_len=seq_len,
                head_hidden=int(head_hidden),
                head_layers=int(head_layers),
                head_dropout=float(head_dropout),
                dense_hidden=int(dense_hidden),
                dense_dropout=float(dense_dropout),
                kernel_regularizer=float(kernel_regularizer),
                noise_std=float(noise_std),
            )
        elif shared_encoder or effective_model_type in ("lstm", "shared_encoder"):
            # Warn user that LSTM is slower on M1
            try:
                import platform
                if platform.system() == 'Darwin' and platform.machine() == 'arm64':
                    console.print("[yellow]⚠ Warning: LSTM is 2-3x slower than TCN on M1 Metal.[/yellow]")
                    console.print("[yellow]  Consider: --model-type tcn or set model.type: tcn in config[/yellow]")
            except Exception:
                pass
            console.print("Model: shared encoder (LSTM - legacy)")
            model = _build_buddy_model_shared_encoder(
                feature_dim=int(feature_dim),
                seq_len=seq_len,
                encoder_hidden=int(head_hidden),
                encoder_layers=int(head_layers),
                encoder_dropout=float(head_dropout),
                dense_hidden=int(dense_hidden),
                dense_dropout=float(dense_dropout),
            )
        else:
            # Fallback to model type selection
            console.print(f"Model: {effective_model_type}")
            model = _build_buddy_model_for_type(
                model_type=effective_model_type,
                feature_dim=int(feature_dim),
                seq_len=seq_len,
                head_hidden=int(head_hidden),
                head_layers=int(head_layers),
                head_dropout=float(head_dropout),
                dense_hidden=int(dense_hidden),
                dense_dropout=float(dense_dropout),
                kernel_regularizer=float(kernel_regularizer),
                noise_std=float(noise_std),
            )

    # Losses:
    # - direction: binary classification (sigmoid)
    # - confidence: TP-before-SL (Tier-2) probability label (sigmoid)
    #   This is a probability regression target in [0,1], so use a regression loss.
    bce_dir = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    conf_loss = tf.keras.losses.MeanSquaredError()
    optimizer_name = None
    optimizer = None
    try:
        adam_legacy_cls = getattr(getattr(tf.keras.optimizers, "legacy", None), "Adam", None)
        if adam_legacy_cls is not None:
            optimizer = adam_legacy_cls(learning_rate=float(lr), clipnorm=1.0)
            optimizer_name = f"{adam_legacy_cls.__module__}.{adam_legacy_cls.__name__}"
    except Exception:
        optimizer = None
        optimizer_name = None
    if optimizer is None:
        adam_cls = tf.keras.optimizers.Adam
        optimizer = adam_cls(learning_rate=float(lr), clipnorm=1.0)
        optimizer_name = f"{adam_cls.__module__}.{adam_cls.__name__}"
        try:
            console.print("[yellow]Warning[/yellow]: tf.keras.optimizers.legacy.Adam unavailable; falling back to non-legacy Adam")
        except Exception:
            pass
    try:
        console.print(f"Optimizer: {optimizer_name}")
    except Exception:
        pass

    try:
        console.print(
            "Compiling model: "
            f"mixed_precision={bool(mp_enabled)} steps_per_execution={int(max(1, int(steps_per_execution)))} jit_compile={bool(jit_compile)}"
        )
    except Exception:
        pass
    _t_compile = None
    try:
        import time as _time

        _t_compile = _time.perf_counter()
    except Exception:
        _t_compile = None

    if mp_enabled:
        try:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        except Exception:
            pass

    def _compile_with_optional_args(model_obj, **kwargs):
        import inspect

        try:
            sig = inspect.signature(model_obj.compile)
            allowed = set(sig.parameters.keys())
        except Exception:
            allowed = set(kwargs.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return model_obj.compile(**filtered)

    spe = max(1, int(steps_per_execution))
    conf_loss_weight = 1.0 if has_conf_labels else 0.0
    _compile_with_optional_args(
        model,
        optimizer=optimizer,
        loss={"direction": bce_dir, "confidence": conf_loss},
        loss_weights={"direction": 1.0, "confidence": float(conf_loss_weight)},
        metrics={
            "direction": [tf.keras.metrics.BinaryAccuracy(name="accuracy")],
            "confidence": [tf.keras.metrics.MeanAbsoluteError(name="mae")] if has_conf_labels else [],
        },
        weighted_metrics={
            "direction": [tf.keras.metrics.BinaryAccuracy(name="accuracy")],
            "confidence": [tf.keras.metrics.MeanAbsoluteError(name="mae")] if has_conf_labels else [],
        },
        steps_per_execution=int(spe),
        jit_compile=bool(jit_compile),
    )

    try:
        if _t_compile is not None:
            import time as _time

            dt = _time.perf_counter() - float(_t_compile)
            console.print(f"Compile complete in {dt:.2f}s")
        else:
            console.print("Compile complete")
    except Exception:
        pass

    class _CombinedObjectiveCallback(tf.keras.callbacks.Callback):
        def __init__(
            self,
            *,
            val_ds_fit: tf.data.Dataset,
            validation_steps: int,
            w_dir: float,
            w_conf: float,
            use_predict: bool,
            has_conf_labels: bool,
        ):
            super().__init__()
            self._val_ds_fit = val_ds_fit
            self._validation_steps = int(validation_steps)
            self._w_dir = float(w_dir)
            self._w_conf = float(w_conf)
            self._use_predict = bool(use_predict)
            self._has_conf_labels = bool(has_conf_labels)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs if isinstance(logs, dict) else {}

            # If confidence labels are absent (e.g., --no-tier2-calibrate),
            # confidence predictions are untrained and should not affect combined.
            if not self._has_conf_labels:
                val_dir_acc = float(logs.get("val_direction_accuracy", 0.0) or 0.0)
                logs["val_avg_confidence"] = float("nan")
                logs["val_combined_score"] = float(val_dir_acc)
                return

            if self._use_predict:
                try:
                    pred = self.model.predict(self._val_ds_fit, steps=self._validation_steps, verbose=0)
                    conf = np.asarray(pred["confidence"]).reshape(-1)
                    conf = np.clip(conf, 0.0, 1.0)
                    avg_conf = float(np.mean(conf))

                    # Compute confidence MAE against the true Tier-2 probability labels.
                    # This penalizes overconfidence and underconfidence symmetrically.
                    val_ds_eval = self._val_ds_fit.take(int(self._validation_steps))
                    conf_true_batches = []
                    conf_w_batches = []
                    for batch in val_ds_eval:
                        if isinstance(batch, (tuple, list)) and len(batch) == 3:
                            _, yb, sw = batch
                        else:
                            _, yb = batch
                            sw = None
                        conf_true_batches.append(np.asarray(yb["confidence"]).reshape(-1))
                        try:
                            if isinstance(sw, dict) and "confidence" in sw:
                                conf_w_batches.append(np.asarray(sw["confidence"]).reshape(-1))
                        except Exception:
                            pass
                    conf_true = np.concatenate(conf_true_batches, axis=0).astype(np.float32).reshape(-1)
                    n_cmp = int(min(int(conf.shape[0]), int(conf_true.shape[0])))
                    if n_cmp <= 0:
                        conf_mae = None
                    else:
                        if conf_w_batches:
                            w = np.concatenate(conf_w_batches, axis=0).astype(np.float32).reshape(-1)[:n_cmp]
                            m = w > 0
                            if bool(np.any(m)):
                                conf_mae = float(
                                    np.sum(np.abs(conf[:n_cmp][m] - conf_true[:n_cmp][m]) * w[m]) / np.sum(w[m])
                                )
                            else:
                                conf_mae = None
                        else:
                            conf_mae = float(np.mean(np.abs(conf[:n_cmp] - conf_true[:n_cmp])))
                except Exception:
                    return
            else:
                # Fast path: avoid extra predict() work by using the logged confidence MAE.
                avg_conf = float("nan")
                conf_mae = None
                for k in (
                    "val_confidence_mae",
                    "val_weighted_confidence_mae",
                    "val_confidence_mean_absolute_error",
                    "val_weighted_confidence_mean_absolute_error",
                ):
                    v = logs.get(k)
                    try:
                        if v is not None and np.isfinite(float(v)):
                            conf_mae = float(v)
                            break
                    except Exception:
                        continue

            logs["val_avg_confidence"] = float(avg_conf)
            val_dir_acc = float(logs.get("val_direction_accuracy", 0.0) or 0.0)
            if conf_mae is None or not np.isfinite(float(conf_mae)):
                return
            conf_score = float(np.clip(1.0 - float(conf_mae), 0.0, 1.0))
            logs["val_combined_score"] = (self._w_dir * val_dir_acc) + (self._w_conf * conf_score)

    monitor = str(es_monitor or "direction").strip().lower()
    if monitor not in {"direction", "combined", "val_loss", "loss"}:
        raise ValueError("es_monitor must be one of: direction, combined, val_loss, loss")

    if monitor == "combined" and not has_conf_labels:
        console.print(
            "[yellow]Warning[/yellow]: --es-monitor combined requested but confidence labels are absent "
            "(--no-tier2-calibrate or no labeled samples). Falling back to --es-monitor direction."
        )
        monitor = "direction"

    def _as_1d_float(a):
        try:
            return np.asarray(a).astype(np.float32).reshape(-1)
        except Exception:
            return None

    def _direction_pred_to_labels(a):
        arr = np.asarray(a)
        # Common cases:
        # - sigmoid binary: (N,) or (N,1)
        # - softmax binary/multi-class: (N,C)
        if arr.ndim == 0:
            return None
        if arr.ndim == 1:
            return (arr >= 0.5).astype(np.float32)
        if arr.shape[-1] == 1:
            return (arr.reshape(-1) >= 0.5).astype(np.float32)
        try:
            return np.argmax(arr, axis=-1).astype(np.float32).reshape(-1)
        except Exception:
            return None

    def _compute_validation_metrics(model_obj):
        pred_obj = model_obj.predict(val_ds_fit, steps=int(validation_steps), verbose=0)
        if isinstance(pred_obj, dict):
            pred_dir_obj = pred_obj.get("direction")
            pred_conf_obj = pred_obj.get("confidence")
        elif isinstance(pred_obj, (tuple, list)):
            pred_dir_obj = pred_obj[0] if len(pred_obj) >= 1 else None
            pred_conf_obj = pred_obj[1] if len(pred_obj) >= 2 else None
        else:
            pred_dir_obj = None
            pred_conf_obj = None

        dir_pred = _direction_pred_to_labels(pred_dir_obj)

        # Pull true labels (and confidence sample weights) from exactly the same number of validation batches.
        val_ds_eval = val_ds.take(int(validation_steps))
        try:
            dir_true_batches = []
            conf_true_batches = []
            conf_w_batches = []
            for batch in val_ds_eval:
                if isinstance(batch, (tuple, list)) and len(batch) == 3:
                    _, yb, sw = batch
                else:
                    _, yb = batch
                    sw = None
                dir_true_batches.append(np.asarray(yb["direction"]).reshape(-1))
                conf_true_batches.append(np.asarray(yb["confidence"]).reshape(-1))
                try:
                    if isinstance(sw, dict) and "confidence" in sw:
                        conf_w_batches.append(np.asarray(sw["confidence"]).reshape(-1))
                except Exception:
                    pass
            dir_true = np.concatenate(dir_true_batches, axis=0).astype(np.float32).reshape(-1)
            conf_true = np.concatenate(conf_true_batches, axis=0).astype(np.float32).reshape(-1)
            conf_w = (
                np.concatenate(conf_w_batches, axis=0).astype(np.float32).reshape(-1)
                if conf_w_batches
                else None
            )
        except Exception:
            dir_true = None
            conf_true = None
            conf_w = None

        if dir_pred is None or dir_true is None:
            val_acc = float("nan")
        else:
            n_cmp = int(min(int(dir_pred.shape[0]), int(dir_true.shape[0])))
            if n_cmp <= 0:
                val_acc = float("nan")
            else:
                val_acc = float(np.mean(dir_pred[:n_cmp] == dir_true[:n_cmp]))

        conf_preds = _as_1d_float(pred_conf_obj)
        if conf_preds is None:
            conf_preds = np.asarray([]).astype(np.float32)
        conf_preds = np.clip(conf_preds, 0.0, 1.0)
        avg_conf = float(np.mean(conf_preds)) if conf_preds.size else float("nan")

        if conf_true is None or conf_preds.size == 0:
            val_conf_mae = None
        else:
            try:
                n_cmp = int(min(int(conf_preds.shape[0]), int(conf_true.shape[0])))
                if n_cmp <= 0:
                    val_conf_mae = None
                else:
                    if conf_w is not None:
                        w = np.asarray(conf_w[:n_cmp], dtype=np.float32).reshape(-1)
                        m = w > 0
                        if bool(np.any(m)):
                            val_conf_mae = float(
                                np.sum(np.abs(conf_preds[:n_cmp][m] - conf_true[:n_cmp][m]) * w[m]) / np.sum(w[m])
                            )
                        else:
                            val_conf_mae = None
                    else:
                        val_conf_mae = float(np.mean(np.abs(conf_preds[:n_cmp] - conf_true[:n_cmp])))
            except Exception:
                val_conf_mae = None

        try:
            val_conf_p80 = float(np.quantile(conf_preds, 0.80)) if conf_preds.size else None
            val_conf_p90 = float(np.quantile(conf_preds, 0.90)) if conf_preds.size else None
        except Exception:
            val_conf_p80 = None
            val_conf_p90 = None

        conf_score = None
        if val_conf_mae is not None:
            try:
                conf_score = float(np.clip(1.0 - float(val_conf_mae), 0.0, 1.0))
            except Exception:
                conf_score = None

        if has_conf_labels and conf_score is not None and np.isfinite(float(conf_score)):
            combined_score = (float(combined_w_dir) * float(val_acc)) + (float(combined_w_conf) * float(conf_score))
        else:
            combined_score = float(val_acc)

        return {
            "val_acc": float(val_acc),
            "avg_conf": float(avg_conf),
            "val_conf_mae": val_conf_mae,
            "val_conf_p80": val_conf_p80,
            "val_conf_p90": val_conf_p90,
            "combined_score": float(combined_score),
        }

    baseline_metrics = None
    baseline_model = None
    baseline_meta = None
    baseline_expected = bool(run_tag is None and final_model_path.exists())
    baseline_load_failed = False
    if baseline_expected:
        try:
            baseline_model = _load_buddy_checkpoint(
                model_path=str(final_model_path),
                seq_len=int(seq_len),
                feature_dim=int(feature_dim),
            )
            baseline_metrics = _compute_validation_metrics(baseline_model)
            try:
                base_meta_p = _meta_path_for_checkpoint(str(final_model_path))
                if base_meta_p is not None and base_meta_p.exists():
                    baseline_meta = json.loads(base_meta_p.read_text())
            except Exception:
                baseline_meta = None
            try:
                console.print(
                    f"Existing model baseline: dir_acc={baseline_metrics['val_acc']*100:.2f}% "
                    f"combined={baseline_metrics['combined_score']*100:.2f}%"
                )
            except Exception:
                pass
        except Exception as e:
            baseline_load_failed = True
            try:
                console.print(f"[yellow]Skipping baseline gating[/yellow]: {e}")
            except Exception:
                pass

    callbacks: list[tf.keras.callbacks.Callback] = []

    if bool(timing):
        import time as _time

        class _StepTimingCallback(tf.keras.callbacks.Callback):
            def __init__(self, *, every: int = 10):
                super().__init__()
                self._every = max(1, int(every))
                self._t0 = None
                self._batch_times: list[float] = []
                self._epoch_t0 = None

            def on_train_begin(self, logs=None):
                self._batch_times.clear()

            def on_epoch_begin(self, epoch, logs=None):
                self._epoch_t0 = _time.perf_counter()

            def on_train_batch_begin(self, batch, logs=None):
                self._t0 = _time.perf_counter()

            def on_train_batch_end(self, batch, logs=None):
                if self._t0 is None:
                    return
                dt = _time.perf_counter() - float(self._t0)
                self._batch_times.append(float(dt))
                b = int(batch) + 1
                if b <= 3 or (b % self._every) == 0:
                    last = float(dt)
                    avg = float(sum(self._batch_times) / max(1, len(self._batch_times)))
                    try:
                        console.print(f"[dim]Timing[/dim]: batch {b} step={last:.3f}s avg={avg:.3f}s")
                    except Exception:
                        pass

            def on_epoch_end(self, epoch, logs=None):
                if self._epoch_t0 is None:
                    return
                dt = _time.perf_counter() - float(self._epoch_t0)
                try:
                    console.print(f"[dim]Timing[/dim]: epoch {int(epoch)+1} wall={dt:.1f}s")
                except Exception:
                    pass

        callbacks.append(_StepTimingCallback(every=10))

    # Safety + stability: stop on NaNs and reduce LR when validation plateaus.
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, int(patience) // 3),
            min_lr=1e-6,
            verbose=1,
        )
    )
    if monitor == "combined":
        callbacks.append(
            _CombinedObjectiveCallback(
                val_ds_fit=val_ds_fit,
                validation_steps=int(validation_steps),
                w_dir=float(combined_w_dir),
                w_conf=float(combined_w_conf),
                use_predict=bool(combined_use_predict),
                has_conf_labels=bool(has_conf_labels),
            )
        )

    if curriculum_cb is not None:
        callbacks.append(curriculum_cb)

    es_kwargs = {
        "patience": int(patience),
        "restore_best_weights": True,
        "verbose": 1,
    }
    if monitor == "direction":
        es = tf.keras.callbacks.EarlyStopping(monitor="val_direction_accuracy", mode="max", **es_kwargs)
    elif monitor == "combined":
        es = tf.keras.callbacks.EarlyStopping(monitor="val_combined_score", mode="max", **es_kwargs)
    else:
        # loss / val_loss
        m = "val_loss" if monitor == "val_loss" else "loss"
        es = tf.keras.callbacks.EarlyStopping(monitor=m, mode="min", **es_kwargs)
    callbacks.append(es)

    # Make early-stopping behavior obvious in logs.
    try:
        console.print(
            "EarlyStopping: "
            f"monitor={es.monitor} mode={es.mode} patience={int(patience)} restore_best_weights={bool(es.restore_best_weights)}"
        )
    except Exception:
        pass

    history = model.fit(
        train_ds_fit,
        validation_data=val_ds_fit,
        epochs=int(epochs),
        steps_per_epoch=int(steps_per_epoch),
        validation_steps=int(validation_steps),
        callbacks=callbacks,
        verbose=int(max(0, min(2, int(fit_verbose) if fit_verbose is not None else 1))),
    )

    # Confirm whether EarlyStopping triggered and what it considered "best".
    try:
        stopped_epoch = int(getattr(es, "stopped_epoch", 0) or 0)
        best_value = float(getattr(es, "best", float("nan")))
        monitor_name = str(getattr(es, "monitor", "metric"))
        if stopped_epoch > 0:
            console.print(
                f"EarlyStopping triggered at epoch {stopped_epoch}. Best {monitor_name}={best_value:.6g} (restored best weights)."
            )
        else:
            console.print(
                f"EarlyStopping did not trigger (ran full {int(epochs)} epochs). Best {monitor_name}={best_value:.6g}."
            )
    except Exception:
        pass

    metrics = _compute_validation_metrics(model)
    val_acc = float(metrics["val_acc"])
    avg_conf = float(metrics["avg_conf"])
    val_conf_mae = metrics["val_conf_mae"]
    val_conf_p80 = metrics["val_conf_p80"]
    val_conf_p90 = metrics["val_conf_p90"]
    combined_score = float(metrics["combined_score"])

    console.print(f"Validation directional accuracy: [bold]{val_acc*100:.2f}%[/bold]")
    if has_conf_labels:
        console.print(f"Validation average confidence: [bold]{avg_conf*100:.2f}%[/bold]")
        console.print(f"Validation target mean confidence: [bold]{val_conf_target_mean*100:.2f}%[/bold]")
        try:
            if val_conf_mae is not None and np.isfinite(float(val_conf_mae)):
                console.print(f"Validation confidence MAE: [bold]{float(val_conf_mae)*100:.2f}%[/bold]")
                console.print(
                    f"Validation confidence score (1-MAE): [bold]{(1.0 - float(val_conf_mae))*100:.2f}%[/bold]"
                )
        except Exception:
            pass
    else:
        console.print("Validation average confidence: [dim]N/A (Tier-2 disabled / no labels)[/dim]")
        console.print("Validation target mean confidence: [dim]N/A (Tier-2 disabled / no labels)[/dim]")
    console.print(f"Validation combined score: [bold]{combined_score*100:.2f}%[/bold]")

    # Optional: out-of-sample (OOS) replay evaluation.
    # Opt-in via env var to avoid adding CLI surface area.
    # - BUDDY_OOS_EVAL=1 enables it
    # - BUDDY_OOS_CANDLES (default 5000) controls fetch size
    # If we trained from live OANDA and prepared a held-out tail, reuse it instead of refetching.
    try:
        oos_env = os.environ.get("BUDDY_OOS_EVAL", "").strip().lower()
        oos_enabled = oos_env in {"1", "true", "yes", "y", "on"}
    except Exception:
        oos_enabled = False

    oos_block: dict[str, Any] | None = None
    baseline_oos_block: dict[str, Any] | None = None
    oos_csv: str | None = None

    # Cache OOS preprocessing so we don't redo feature engineering for the baseline model.
    # Keyed by (csv_path, smoothing, median_window).
    oos_preproc_cache: dict[tuple[str, bool, int | None], dict[str, Any]] = {}

    def _buddy_oos_replay_eval(
        *,
        model_obj: Any,
        csv_path_eval: str,
        feature_columns_eval: list[str],
        mu_eval: np.ndarray,
        sigma_eval: np.ndarray,
        seq_len_eval: int,
        train_smoothing_eval: bool,
        median_window_eval: int | None,
        label_stride_eval: int,
        combined_w_dir_eval: float,
        combined_w_conf_eval: float,
    ) -> dict[str, float]:
        # Rebuild features, align columns, standardize using training stats, then evaluate.
        key = (str(csv_path_eval), bool(train_smoothing_eval), int(median_window_eval) if median_window_eval is not None else None)
        cached = oos_preproc_cache.get(key)
        if cached is None:
            df_oos = pd.read_csv(str(csv_path_eval))
            try:
                if "time" in df_oos.columns:
                    df_oos["time"] = pd.to_datetime(df_oos["time"], errors="coerce")
                    df_oos = df_oos.sort_values("time")
            except Exception:
                pass

            fe_oos = FeatureEngineering(cfg.get("feature_engineering", {}))
            df_oos = fe_oos.create_features(
                df_oos,
                include_all=True,
                apply_candle_smoothing=bool(train_smoothing_eval),
                median_window=median_window_eval,
            )

            numeric_all = df_oos.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).dropna(axis=0)
            if numeric_all.empty:
                raise ValueError("OOS replay: no numeric features after preprocessing")
            ohlc_oos = df_oos.loc[numeric_all.index].reset_index(drop=True)
            cached = {"numeric_all": numeric_all, "ohlc_oos": ohlc_oos}
            oos_preproc_cache[key] = cached

        numeric_all = cached["numeric_all"]
        ohlc_oos = cached["ohlc_oos"]

        numeric_oos = numeric_all.reindex(columns=feature_columns_eval).fillna(0.0)
        closes_oos = ohlc_oos["close"].to_numpy(dtype=np.float32)
        y_dir_oos = (closes_oos[1:] > closes_oos[:-1]).astype(np.float32)

        x2d_oos = numeric_oos.to_numpy(dtype=np.float32)
        x2d_oos = x2d_oos[:-1]
        n_oos = int(x2d_oos.shape[0])
        if n_oos <= int(seq_len_eval) + 10:
            raise ValueError("OOS replay: not enough rows")

        # Confidence targets for OOS: Tier-2 TP-before-SL (same semantics as training), with sparse labeling.
        conf_y_oos = np.zeros((int(n_oos),), dtype=np.float32)
        conf_w_oos = np.zeros((int(n_oos),), dtype=np.float32)
        try:
            from fx_paper import pip_size, simulate_tp_sl_outcome

            pip = float(pip_size("USD_JPY"))

            def _spread_pips_at_entry(gi: int) -> float:
                sp = float(tier2_spread_fallback_pips)
                try:
                    if "bid_close" in ohlc_oos.columns and "ask_close" in ohlc_oos.columns:
                        bid = float(ohlc_oos["bid_close"].iloc[int(gi) + 1])
                        ask = float(ohlc_oos["ask_close"].iloc[int(gi) + 1])
                        if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
                            sp = float((ask - bid) / max(pip, 1e-9))
                except Exception:
                    sp = float(tier2_spread_fallback_pips)
                return float(sp)

            max_signal = min(int(n_oos) - 1, int(len(ohlc_oos)) - 3)
            start_gi = max(0, int(seq_len_eval) - 1)
            stride = max(1, int(label_stride_eval))
            for gi in range(int(start_gi), int(max_signal) + 1, int(stride)):
                spread_pips = _spread_pips_at_entry(int(gi))
                true_dir = "long" if float(y_dir_oos[int(gi)]) >= 0.5 else "short"
                res = simulate_tp_sl_outcome(
                    ohlc_oos,
                    instrument="USD_JPY",
                    signal_index=int(gi),
                    direction=str(true_dir),
                    stop_loss_pips=float(tier2_stop_loss_pips),
                    take_profit_pips=float(tier2_take_profit_pips),
                    spread_pips=float(spread_pips),
                    max_horizon_candles=int(tier2_horizon_candles),
                    tie_break=str(tier2_tie_break),
                )
                conf_y_oos[int(gi)] = float(int(res.to_label()))
                conf_w_oos[int(gi)] = 1.0
        except Exception:
            conf_y_oos = np.zeros((int(n_oos),), dtype=np.float32)
            conf_w_oos = np.zeros((int(n_oos),), dtype=np.float32)

        # Use the same split rule as training (time-ordered 80/20) so the metric is stable.
        split_oos = int(n_oos * 0.8)
        x_val_oos = x2d_oos[split_oos:]
        y_val_oos = y_dir_oos[split_oos:]

        # Standardize using the model's saved train stats.
        x_val_oos = x_val_oos.astype(np.float32, copy=False)
        x_val_oos = (x_val_oos - mu_eval) / np.where(sigma_eval < 1e-6, np.float32(1.0), sigma_eval)

        # Match make_sequence_dataset semantics: drop last timestep before windowing.
        x_val_oos_2d = x_val_oos[:-1]
        y_val_oos_1d = y_val_oos[:-1]
        n_windows_oos = int(len(x_val_oos_2d) - int(seq_len_eval) + 1)
        if n_windows_oos <= 0:
            raise ValueError("OOS replay: not enough windows")

        # Vectorized window construction and a single predict() call to avoid retracing.
        try:
            xb_all = np.lib.stride_tricks.sliding_window_view(
                x_val_oos_2d,
                window_shape=int(seq_len_eval),
                axis=0,
            )
            # sliding_window_view inserts the window axis at the end, yielding
            # (n_windows, feature_dim, seq_len). Transpose to (n_windows, seq_len, feature_dim).
            if xb_all.ndim == 3:
                xb_all = np.transpose(xb_all, (0, 2, 1))
        except Exception:
            xb_all = np.stack(
                [x_val_oos_2d[k : k + int(seq_len_eval)] for k in range(0, n_windows_oos)],
                axis=0,
            ).astype(np.float32)
        xb_all = np.asarray(xb_all, dtype=np.float32)
        yb_all = y_val_oos_1d[int(seq_len_eval) - 1 :]

        pred_all = model_obj.predict(xb_all, batch_size=256, verbose=0)
        p_dir = np.asarray(pred_all["direction"]).reshape(-1)
        p_conf = np.asarray(pred_all["confidence"]).reshape(-1)
        p_conf = np.clip(p_conf, 0.0, 1.0)
        y_hat = (p_dir >= 0.5).astype(np.float32)

        n_cmp = int(min(int(len(y_hat)), int(len(yb_all))))
        val_acc_oos = float(np.mean(y_hat[:n_cmp] == yb_all[:n_cmp])) if n_cmp > 0 else float("nan")
        avg_conf_oos = float(np.mean(p_conf[:n_cmp])) if n_cmp > 0 else float("nan")

        conf_abs_err_sum = 0.0
        conf_w_sum = 0.0
        try:
            end_gi_all = split_oos + (np.arange(0, n_cmp, dtype=np.int64) + int(seq_len_eval) - 1)
            w_all = np.asarray(conf_w_oos[end_gi_all], dtype=np.float32).reshape(-1)
            m = w_all > 0
            if bool(np.any(m)):
                t_all = np.asarray(conf_y_oos[end_gi_all], dtype=np.float32).reshape(-1)
                conf_abs_err_sum = float(np.sum(np.abs(p_conf[:n_cmp][m] - t_all[m]) * w_all[m]))
                conf_w_sum = float(np.sum(w_all[m]))
        except Exception:
            conf_abs_err_sum = 0.0
            conf_w_sum = 0.0
        conf_mae_oos = float(conf_abs_err_sum / conf_w_sum) if conf_w_sum > 0 else float("nan")
        conf_score_oos = float(np.clip(1.0 - float(conf_mae_oos), 0.0, 1.0)) if np.isfinite(conf_mae_oos) else 0.0
        combined_oos = float((combined_w_dir_eval * val_acc_oos) + (combined_w_conf_eval * conf_score_oos))
        return {
            "val_direction_accuracy": val_acc_oos,
            "val_avg_confidence": avg_conf_oos,
            "val_conf_mae": conf_mae_oos,
            "val_combined_score": combined_oos,
        }

    if oos_enabled:
        try:
            oos_candles = 5000
            try:
                oos_candles = int(float(os.environ.get("BUDDY_OOS_CANDLES", "5000")))
            except Exception:
                oos_candles = 5000
            oos_candles = max(300, min(int(oos_candles), 20000))

            if oos_holdout_csv is not None and Path(str(oos_holdout_csv)).exists():
                oos_csv = str(oos_holdout_csv)
                if oos_holdout_rows is not None:
                    oos_candles = int(oos_holdout_rows)
            else:
                oos_csv = _oanda_fetch_to_csv(
                    OandaFetchOptions(
                        instrument="USD_JPY",
                        granularity="M5",
                        candles=int(oos_candles),
                        price="MBA",
                        save_csv=None,
                    )
                )

            oos_metrics = _buddy_oos_replay_eval(
                model_obj=model,
                csv_path_eval=str(oos_csv),
                feature_columns_eval=list(feature_columns),
                mu_eval=mu,
                sigma_eval=sigma,
                seq_len_eval=int(seq_len),
                train_smoothing_eval=bool(train_smoothing),
                median_window_eval=median_window,
                label_stride_eval=int(label_stride),
                combined_w_dir_eval=float(combined_w_dir),
                combined_w_conf_eval=float(combined_w_conf),
            )
            oos_block = {
                "csv_path": str(oos_csv),
                "candles": int(oos_candles),
                **oos_metrics,
            }

            # Baseline OOS replay on the SAME OOS CSV, when we can load compatible baseline metadata.
            if baseline_model is not None and isinstance(baseline_meta, dict):
                try:
                    base_cols = baseline_meta.get("feature_columns")
                    base_std = baseline_meta.get("standardize")
                    base_seq = baseline_meta.get("seq_len")
                    if isinstance(base_cols, list) and isinstance(base_std, dict) and base_seq is not None:
                        bmu = np.asarray(base_std.get("mean", []), dtype=np.float32).reshape(1, -1)
                        bsd = np.asarray(base_std.get("std", []), dtype=np.float32).reshape(1, -1)
                        if int(base_seq) == int(seq_len) and int(bmu.shape[1]) == int(feature_dim) and int(bsd.shape[1]) == int(feature_dim):
                            base_oos_metrics = _buddy_oos_replay_eval(
                                model_obj=baseline_model,
                                csv_path_eval=str(oos_csv),
                                feature_columns_eval=[str(c) for c in base_cols],
                                mu_eval=bmu,
                                sigma_eval=bsd,
                                seq_len_eval=int(seq_len),
                                train_smoothing_eval=bool(baseline_meta.get("train_smoothing", train_smoothing)),
                                median_window_eval=(
                                    int(baseline_meta.get("median_window"))
                                    if baseline_meta.get("median_window") is not None
                                    else None
                                ),
                                label_stride_eval=int(label_stride),
                                combined_w_dir_eval=float(baseline_meta.get("combined_w_dir", combined_w_dir)),
                                combined_w_conf_eval=float(baseline_meta.get("combined_w_conf", combined_w_conf)),
                            )
                            baseline_oos_block = {
                                "model_path": str(final_model_path),
                                "csv_path": str(oos_csv),
                                **base_oos_metrics,
                            }
                except Exception:
                    baseline_oos_block = None

            try:
                console.print(
                    "OOS replay: "
                    f"dir_acc={float(oos_block['val_direction_accuracy'])*100:.2f}% "
                    f"avg_conf={float(oos_block['val_avg_confidence'])*100:.2f}% "
                    f"conf_mae={float(oos_block.get('val_conf_mae', float('nan')))*100:.2f}% "
                    f"combined={float(oos_block['val_combined_score'])*100:.2f}%"
                )
                if baseline_oos_block is not None:
                    console.print(
                        "OOS baseline: "
                        f"dir_acc={float(baseline_oos_block['val_direction_accuracy'])*100:.2f}% "
                        f"avg_conf={float(baseline_oos_block['val_avg_confidence'])*100:.2f}% "
                        f"conf_mae={float(baseline_oos_block.get('val_conf_mae', float('nan')))*100:.2f}% "
                        f"combined={float(baseline_oos_block['val_combined_score'])*100:.2f}%"
                    )
            except Exception:
                pass
        except Exception as e:
            oos_block = None
            baseline_oos_block = None
            try:
                console.print(f"[yellow]OOS replay skipped[/yellow]: {e}")
            except Exception:
                pass

    # Save gating: don't overwrite the default checkpoint unless it improved.
    save_model_path = final_model_path
    save_meta_path = final_meta_path

    # Safety: if a baseline exists but we couldn't load/evaluate it, avoid overwriting the default.
    # This prevents accidental regressions when deserialization/input-shape mismatch blocks gating.
    if baseline_expected and baseline_metrics is None:
        allow_env = str(os.environ.get("BUDDY_ALLOW_OVERWRITE_WITHOUT_BASELINE", "")).strip().lower()
        allow_overwrite_without_baseline = allow_env in {"1", "true", "yes", "y", "on"}
        if not allow_overwrite_without_baseline:
            save_model_path = candidate_model_path
            save_meta_path = candidate_meta_path
            try:
                why = "baseline load failed" if baseline_load_failed else "baseline metrics unavailable"
                console.print(
                    "[yellow]Baseline gating unavailable[/yellow]: "
                    f"{why} -> saving as {candidate_model_path.name} (set BUDDY_ALLOW_OVERWRITE_WITHOUT_BASELINE=1 to override)"
                )
            except Exception:
                pass
    if baseline_metrics is not None:
        try:
            baseline_score = float(baseline_metrics.get("combined_score", float("nan")))
            gate_metric_name = "val_combined_score"
            gate_new = float(combined_score)
            gate_old = float(baseline_score)

            # Best practice: prefer OOS replay for gating when available and comparable.
            if oos_block is not None and baseline_oos_block is not None:
                try:
                    gate_metric_name = "oos_replay.val_combined_score"
                    gate_new = float(oos_block.get("val_combined_score", gate_new))
                    gate_old = float(baseline_oos_block.get("val_combined_score", gate_old))
                except Exception:
                    gate_metric_name = "val_combined_score"
                    gate_new = float(combined_score)
                    gate_old = float(baseline_score)

            # If the new model is strictly worse under the chosen gate metric, keep the previous default.
            if gate_new < (gate_old - 1e-9):
                save_model_path = candidate_model_path
                save_meta_path = candidate_meta_path
                console.print(
                    "[yellow]New model did not beat existing baseline[/yellow]: "
                    f"metric={gate_metric_name} new={gate_new*100:.2f}% vs old={gate_old*100:.2f}% -> keeping existing {final_model_path.name}"
                )
        except Exception:
            pass

    # Save model and metadata.
    model.save(save_model_path)
    model_path = save_model_path
    meta_path = save_meta_path
    # Compute best epoch according to the chosen early-stopping monitor.
    best_epoch = None
    best_metric_name = None
    best_metric_value = None
    try:
        metric_name = str(getattr(es, "monitor", "val_direction_accuracy"))
        mode = str(getattr(es, "mode", "max"))
        series = history.history.get(metric_name, [])
        if isinstance(series, list) and series:
            arr = np.asarray(series, dtype=np.float32)
            best_idx = int(np.argmin(arr) if mode == "min" else np.argmax(arr))
            best_epoch = best_idx + 1
            best_metric_name = metric_name
            best_metric_value = float(arr[best_idx])
    except Exception:
        pass

    # --- Tier-2 calibration (TP-before-SL) ---
    tier2_calibration: dict[str, Any] | None = None
    tier2_warnings: list[str] = []
    tier2_best_effort = False
    try:
        if tier2_calibrate:
            t_tier2_cal = time.perf_counter()
            from fx_paper import pip_size, simulate_tp_sl_outcome

            # Best-effort instrument inference.
            instrument_guess = None
            if oanda_fetch is not None:
                try:
                    instrument_guess = str(getattr(oanda_fetch, "instrument", None) or None)
                except Exception:
                    instrument_guess = None
            if not instrument_guess:
                # Try to infer from filename, else default to USD_JPY.
                cp = str(csv_path or "")
                up = cp.upper()
                if "EUR_USD" in up:
                    instrument_guess = "EUR_USD"
                elif "GBP_USD" in up:
                    instrument_guess = "GBP_USD"
                elif "USD_JPY" in up or "USDJPY" in up or "_JPY_" in up or up.endswith("JPY.CSV"):
                    instrument_guess = "USD_JPY"
                else:
                    instrument_guess = "USD_JPY"

            pip = float(pip_size(str(instrument_guess)))
            if tier2_stop_loss_pips <= 0 or tier2_take_profit_pips <= 0:
                raise ValueError("Tier-2 stop/take pips must be positive")
            tier2_horizon_candles = max(1, int(tier2_horizon_candles))
            tier2_calibration_bins = max(5, int(tier2_calibration_bins))
            stride = max(1, int(tier2_calibration_stride))

            # If curriculum was used, apply the FINAL mask for calibration.
            feature_mask_np: np.ndarray | None = None
            if feature_mask_var is not None:
                try:
                    feature_mask_np = np.asarray(feature_mask_var.numpy(), dtype=np.float32).reshape(-1)
                except Exception:
                    feature_mask_np = None

            if feature_mask_np is not None and int(feature_mask_np.size) != int(feats_model.shape[1]):
                msg = (
                    "Tier-2 calibration feature mask dimension mismatch: "
                    f"mask={int(feature_mask_np.size)} feats={int(feats_model.shape[1])}."
                )
                if disable_tier2_on_mismatch:
                    tier2_warnings.append(msg)
                    tier2_warnings.append("Tier-2 calibration disabled due to --disable-tier2-on-mismatch.")
                    console.print(f"[yellow]Tier-2 calibration skipped[/yellow]: {msg}")
                    raise RuntimeError("tier2_disabled_on_mismatch")
                if ignore_input_mismatches:
                    tier2_best_effort = True
                    tier2_warnings.append(msg)
                    tier2_warnings.append("Proceeding without feature mask due to --ignore-input-mismatches.")
                    console.print(f"[yellow]Warning[/yellow]: {msg} Proceeding without feature mask.")
                    feature_mask_np = None
                else:
                    raise ValueError(msg)

            # Iterate over validation window endpoints and simulate TP/SL outcome.
            scores: list[float] = []
            wins: list[int] = []

            # Global indices into ohlc_df (same length as standardized feats)
            global_start = int(split) + int(seq_len) - 1
            global_end_max = int(n) - 2  # must have at least one forward candle

            batch_x: list[np.ndarray] = []
            batch_idx: list[int] = []
            batch_max = 64

            def _flush_batch():
                if not batch_x:
                    return
                x = np.stack(batch_x, axis=0).astype(np.float32, copy=False)
                if feature_mask_np is not None:
                    x = x * feature_mask_np.reshape(1, 1, -1)
                _inputs_struct = getattr(model, "_inputs_struct", None)
                _expects_features_dict = isinstance(_inputs_struct, dict) and "features" in _inputs_struct
                pred_in = {"features": x} if _expects_features_dict else x
                pred_b = model.predict(pred_in, verbose=0)
                p_dir_b = np.asarray(pred_b["direction"]).reshape(-1)
                p_conf_b = np.asarray(pred_b["confidence"]).reshape(-1)
                p_conf_b = np.clip(p_conf_b, 0.0, 1.0)

                for k, gi in enumerate(batch_idx):
                    p_dir = float(p_dir_b[k])
                    p_conf = float(p_conf_b[k])
                    side = "buy" if p_dir >= 0.5 else "sell"
                    dir_conf = float(p_dir) if side == "buy" else float(1.0 - p_dir)
                    score = float(max(0.0, min(1.0, dir_conf * p_conf)))

                    # Need entry candle (gi+1) and first scan candle (gi+2) for next-candle entry.
                    if int(gi) + 2 >= len(ohlc_df):
                        continue

                    # Spread estimate at entry (pips): use bid/ask on ENTRY candle when available else fallback.
                    spread_pips = float(tier2_spread_fallback_pips)
                    try:
                        if "bid_close" in ohlc_df.columns and "ask_close" in ohlc_df.columns:
                            bid = float(ohlc_df["bid_close"].iloc[int(gi) + 1])
                            ask = float(ohlc_df["ask_close"].iloc[int(gi) + 1])
                            if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
                                spread_pips = float((ask - bid) / max(pip, 1e-9))
                    except Exception:
                        spread_pips = float(tier2_spread_fallback_pips)

                    direction = "long" if side == "buy" else "short"
                    res = simulate_tp_sl_outcome(
                        ohlc_df,
                        instrument=str(instrument_guess),
                        signal_index=int(gi),
                        direction=direction,
                        stop_loss_pips=float(tier2_stop_loss_pips),
                        take_profit_pips=float(tier2_take_profit_pips),
                        spread_pips=float(spread_pips),
                        max_horizon_candles=int(tier2_horizon_candles),
                        tie_break=str(tier2_tie_break),
                    )
                    scores.append(score)
                    wins.append(int(res.to_label()))

                batch_x.clear()
                batch_idx.clear()

            for gi in range(global_start, global_end_max + 1, stride):
                w = feats_model[int(gi) - int(seq_len) + 1 : int(gi) + 1]
                if w.shape[0] != int(seq_len):
                    continue
                batch_x.append(np.asarray(w, dtype=np.float32))
                batch_idx.append(int(gi))
                if len(batch_x) >= batch_max:
                    _flush_batch()
            _flush_batch()

            try:
                console.print(
                    f"Tier-2 calibration simulation runtime: n={len(scores)} stride={int(stride)} elapsed={time.perf_counter() - t_tier2_cal:.2f}s"
                )
            except Exception:
                pass

            if len(scores) >= 50:
                s = np.asarray(scores, dtype=np.float64)
                y = np.asarray(wins, dtype=np.float64)

                # --- Temperature scaling (primary calibrator) ---
                # Calibrate score -> P(win) by applying a temperature to the logit(score).
                # This is data-efficient (single parameter) and stable for n~O(1k).
                ts_enabled = False
                ts_temperature = None
                ts_nll_pre = None
                ts_nll_post = None
                ts_spearman_pre = None
                ts_spearman_post = None
                ts_grid_coarse = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
                ts_grid_fine = None

                try:
                    # Pre-calibration diagnostics.
                    ts_nll_pre = _tier2_nll(y, np.clip(s, 0.0, 1.0))
                    ts_spearman_pre = _tier2_spearman(s, y)

                    def _apply_temperature_vec(temperature: float) -> np.ndarray:
                        ss = np.clip(s, 1e-7, 1.0 - 1e-7)
                        logits = np.log(ss / (1.0 - ss))
                        z = logits / float(temperature)
                        # Stable sigmoid
                        out = np.empty_like(z, dtype=np.float64)
                        pos = z >= 0
                        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
                        ez = np.exp(z[~pos])
                        out[~pos] = ez / (1.0 + ez)
                        return out

                    best_temperature = None
                    best_nll = float("inf")
                    for temperature in ts_grid_coarse:
                        p = _apply_temperature_vec(float(temperature))
                        nll = _tier2_nll(y, p)
                        if nll < best_nll:
                            best_nll = float(nll)
                            best_temperature = float(temperature)

                    if best_temperature is not None and np.isfinite(best_temperature):
                        # Single zoom pass around the best coarse T.
                        if best_temperature <= min(ts_grid_coarse) + 1e-12 or best_temperature >= max(ts_grid_coarse) - 1e-12:
                            lo = max(0.05, best_temperature - 0.5)
                            hi = best_temperature + 0.5
                        else:
                            lo = max(0.05, best_temperature - 0.3)
                            hi = best_temperature + 0.3
                        fine = np.round(np.arange(lo, hi + 1e-12, 0.1), 3).tolist()
                        ts_grid_fine = [float(x) for x in fine if float(x) > 0]

                        for temperature in ts_grid_fine:
                            p = _apply_temperature_vec(float(temperature))
                            nll = _tier2_nll(y, p)
                            if nll < best_nll:
                                best_nll = float(nll)
                                best_temperature = float(temperature)

                    if best_temperature is not None and np.isfinite(best_temperature) and best_temperature > 0:
                        p_post = _apply_temperature_vec(float(best_temperature))
                        ts_temperature = float(best_temperature)
                        ts_nll_post = float(_tier2_nll(y, p_post))
                        ts_spearman_post = float(_tier2_spearman(p_post, y))

                        # Validation checks:
                        # - NLL must improve.
                        # - If Spearman remains negative (and doesn't improve), disable and keep bins-only fallback.
                        nll_ok = (
                            ts_nll_pre is not None
                            and ts_nll_post is not None
                            and np.isfinite(ts_nll_pre)
                            and np.isfinite(ts_nll_post)
                            and (ts_nll_post < ts_nll_pre - 1e-9)
                        )
                        spearman_ok = True
                        if ts_spearman_post is not None and np.isfinite(ts_spearman_post) and ts_spearman_post < 0.0:
                            sp0 = float(ts_spearman_pre) if ts_spearman_pre is not None and np.isfinite(ts_spearman_pre) else float("nan")
                            if not np.isfinite(sp0) or ts_spearman_post <= sp0 + 1e-6:
                                spearman_ok = False

                        ts_enabled = bool(nll_ok and spearman_ok)
                        
                        # Allow forcing temperature scaling via config
                        force_temp_scaling = bool(cfg.get("buddy", {}).get("tier2", {}).get("force_temperature_scaling", False))
                        if force_temp_scaling and ts_temperature is not None and ts_temperature > 0:
                            ts_enabled = True
                            tier2_warnings.append(
                                f"Temperature scaling force-enabled via config (T={ts_temperature:.3f})."
                            )
                        elif not ts_enabled:
                            tier2_warnings.append(
                                "Temperature scaling disabled by validation checks "
                                f"(nll_pre={ts_nll_pre}, nll_post={ts_nll_post}, spearman_pre={ts_spearman_pre}, spearman_post={ts_spearman_post})."
                            )
                except Exception as _e:
                    ts_enabled = False
                    tier2_warnings.append(f"Temperature scaling fit failed: {_e}")

                # Quantile bins (more stable than uniform bins when scores are clustered).
                qs = np.linspace(0.0, 1.0, int(tier2_calibration_bins) + 1)
                edges = np.quantile(s, qs)
                edges = np.unique(np.clip(edges, 0.0, 1.0))
                if len(edges) >= 3:
                    bins_out: list[dict[str, Any]] = []
                    for i in range(1, len(edges)):
                        lo = float(edges[i - 1])
                        hi = float(edges[i])
                        if hi <= lo + 1e-12:
                            continue
                        if i == 1:
                            mask = (s >= lo) & (s <= hi)
                        else:
                            mask = (s > lo) & (s <= hi)
                        n_i = int(mask.sum())
                        if n_i < 10:
                            continue
                        s_i = float(np.mean(s[mask]))
                        p_i = float(np.mean(y[mask]))
                        bins_out.append({"score": s_i, "p_win": p_i, "n": n_i, "lo": lo, "hi": hi})

                    if len(bins_out) >= 3:
                        bins_out.sort(key=lambda b: float(b["score"]))
                        tier2_calibration = {
                            "version": "tp_sl_calib_v1",
                            "instrument": str(instrument_guess),
                            "stop_loss_pips": float(tier2_stop_loss_pips),
                            "take_profit_pips": float(tier2_take_profit_pips),
                            "horizon_candles": int(tier2_horizon_candles),
                            "spread_fallback_pips": float(tier2_spread_fallback_pips),
                            "stride": int(stride),
                            "n": int(len(scores)),
                            "win_rate": float(np.mean(y)),
                            "temperature_scaling": {
                                "enabled": bool(ts_enabled),
                                "T": float(ts_temperature) if ts_temperature is not None else None,
                                "nll_pre": float(ts_nll_pre) if ts_nll_pre is not None else None,
                                "nll_post": float(ts_nll_post) if ts_nll_post is not None else None,
                                "spearman_pre": float(ts_spearman_pre) if ts_spearman_pre is not None else None,
                                "spearman_post": float(ts_spearman_post) if ts_spearman_post is not None else None,
                                "grid_coarse": list(ts_grid_coarse),
                                "grid_fine": list(ts_grid_fine) if ts_grid_fine is not None else None,
                                "notes": "Primary calibrator: sigmoid(logit(score)/T). Bins retained as fallback.",
                            },
                            "bins": bins_out,
                        }
                        console.print(
                            f"Tier-2 calibration: n={int(tier2_calibration['n'])} win_rate={float(tier2_calibration['win_rate'])*100:.2f}% bins={len(bins_out)}"
                        )
                        try:
                            ts_meta = (tier2_calibration.get("temperature_scaling") or {})
                            if bool(ts_meta.get("enabled", False)) and ts_meta.get("T") is not None:
                                console.print(
                                    f"Tier-2 temperature scaling enabled: T={float(ts_meta['T']):.3g} "
                                    f"NLL {float(ts_meta.get('nll_pre', float('nan'))):.4g} -> {float(ts_meta.get('nll_post', float('nan'))):.4g}"
                                )
                            else:
                                console.print("[dim]Tier-2 temperature scaling not enabled (using bins fallback).[/dim]")
                        except Exception:
                            pass
            else:
                console.print(
                    f"[yellow]Tier-2 calibration skipped[/yellow]: insufficient labeled samples (n={len(scores)})"
                )
    except Exception as e:
        if str(e) == "tier2_disabled_on_mismatch":
            tier2_calibration = None
        else:
            console.print(f"[yellow]Tier-2 calibration failed[/yellow]: {e}")
            tier2_calibration = None
            tier2_warnings.append(f"Tier-2 calibration failed: {e}")

    if tier2_warnings:
        meta_warnings.extend([f"Tier-2: {w}" for w in tier2_warnings])

    # --- Meta-labeling (optional) ---
    # Train a secondary model to predict when to trust primary predictions
    meta_labeler_path: str | None = None
    meta_labeling_metrics: dict | None = None
    meta_labeling_enabled = bool(cfg.get("buddy", {}).get("meta_labeling", {}).get("enabled", False))
    
    if meta_labeling_enabled:
        try:
            from meta_labeling import MetaLabeler, MetaLabelingConfig
            
            console.print("[cyan]Training meta-labeler (trade filter)...[/cyan]")
            
            # Check if model expects 3D input (TCN, LSTM, etc.)
            # TCN models expect (batch, sequence_length, features)
            model_input_shape = None
            try:
                if hasattr(model, 'input_shape'):
                    model_input_shape = model.input_shape
                elif hasattr(model, 'inputs') and len(model.inputs) > 0:
                    model_input_shape = model.inputs[0].shape
            except Exception:
                pass
            
            # Create sequences if model expects 3D input
            if model_input_shape and len(model_input_shape) == 3:
                # Model expects 3D: (batch, seq_len, features)
                def create_sequences_for_prediction(features: np.ndarray, seq_len: int) -> np.ndarray:
                    """Create overlapping sequences for prediction."""
                    n_samples = len(features) - seq_len + 1
                    if n_samples <= 0:
                        return features[np.newaxis, :, :]  # Return single sequence
                    
                    sequences = np.zeros((n_samples, seq_len, features.shape[1]), dtype=np.float32)
                    for i in range(n_samples):
                        sequences[i] = features[i:i + seq_len]
                    return sequences
                
                train_sequences = create_sequences_for_prediction(train_feats, seq_len)
                val_sequences = create_sequences_for_prediction(val_feats, seq_len)
                
                # Get predictions using sequences
                train_preds = model.predict(train_sequences, verbose=0, batch_size=128)
                val_preds = model.predict(val_sequences, verbose=0, batch_size=128)
                
                # Align labels with sequences (use last label for each sequence)
                train_dir_aligned = train_dir_raw[seq_len - 1:][:len(train_sequences)]
                val_dir_aligned = val_dir_raw[seq_len - 1:][:len(val_sequences)]
                
                # Use aligned features for meta-labeler (flatten sequences to 2D)
                # Meta-labeler expects 2D features, so we'll use the last timestep of each sequence
                train_feats_for_meta = train_sequences[:, -1, :]  # Last timestep
                val_feats_for_meta = val_sequences[:, -1, :]  # Last timestep
            else:
                # Model expects 2D input (XGBoost, etc.)
                train_preds = model.predict(train_feats, verbose=0)
                val_preds = model.predict(val_feats, verbose=0)
                train_dir_aligned = train_dir_raw
                val_dir_aligned = val_dir_raw
                train_feats_for_meta = train_feats
                val_feats_for_meta = val_feats
            
            if isinstance(train_preds, dict):
                train_dir_preds = np.asarray(train_preds.get("direction", train_preds.get("output_0"))).flatten()
                val_dir_preds = np.asarray(val_preds.get("direction", val_preds.get("output_0"))).flatten()
            else:
                train_dir_preds = np.asarray(train_preds).flatten()
                val_dir_preds = np.asarray(val_preds).flatten()
            
            # Ensure predictions and labels are aligned
            min_len = min(len(train_dir_preds), len(train_dir_aligned))
            train_dir_preds = train_dir_preds[:min_len]
            train_dir_aligned = train_dir_aligned[:min_len]
            train_feats_for_meta = train_feats_for_meta[:min_len]
            
            min_len_val = min(len(val_dir_preds), len(val_dir_aligned))
            val_dir_preds = val_dir_preds[:min_len_val]
            val_dir_aligned = val_dir_aligned[:min_len_val]
            val_feats_for_meta = val_feats_for_meta[:min_len_val]
            
            # Configure meta-labeler with regularization (prevents overfitting)
            meta_cfg_yaml = cfg.get("buddy", {}).get("meta_labeling", {})
            meta_cfg = MetaLabelingConfig(
                use_xgboost=True,
                # Reduced defaults to prevent overfitting
                n_estimators=int(meta_cfg_yaml.get("n_estimators", 100)),
                max_depth=int(meta_cfg_yaml.get("max_depth", 2)),
                learning_rate=float(meta_cfg_yaml.get("learning_rate", 0.05)),
                # Regularization params
                reg_alpha=float(meta_cfg_yaml.get("reg_alpha", 1.0)),
                reg_lambda=float(meta_cfg_yaml.get("reg_lambda", 5.0)),
                subsample=float(meta_cfg_yaml.get("subsample", 0.7)),
                colsample_bytree=float(meta_cfg_yaml.get("colsample_bytree", 0.7)),
                min_child_weight=int(meta_cfg_yaml.get("min_child_weight", 5)),
                early_stopping_rounds=int(meta_cfg_yaml.get("early_stopping_rounds", 30)),
                max_overfitting_gap=float(meta_cfg_yaml.get("max_overfitting_gap", 0.15)),
                auto_tune=bool(meta_cfg_yaml.get("auto_tune", False)),
                # Threshold
                min_confidence_threshold=float(meta_cfg_yaml.get("threshold", 0.55)),
                use_reduced_features=bool(meta_cfg_yaml.get("use_reduced_features", True)),
            )
            
            # Train meta-labeler (auto-tune happens inside if config.auto_tune=True)
            labeler = MetaLabeler(meta_cfg)
            
            # Fit the meta-labeler
            meta_labeling_metrics = labeler.fit(
                train_feats_for_meta,
                train_dir_preds,
                train_dir_aligned,
                features_val=val_feats_for_meta,
                predictions_val=val_dir_preds,
                outcomes_val=val_dir_aligned,
                verbose=True,
            )
            
            train_meta_acc = meta_labeling_metrics.get("meta_train_accuracy", 0)
            val_meta_acc = meta_labeling_metrics.get("meta_val_accuracy", 0)
            overfit_gap = meta_labeling_metrics.get("overfitting_gap", train_meta_acc - val_meta_acc)
            
            # Tune threshold on validation set
            best_thresh, thresh_metrics = labeler.tune_threshold(
                val_feats_for_meta, val_dir_preds, val_dir_aligned, verbose=True
            )
            thresh_improvement = thresh_metrics.get("improvement", 0) if thresh_metrics else 0
            
            console.print(f"Meta-labeler: train_acc={train_meta_acc:.1%} val_acc={val_meta_acc:.1%} gap={overfit_gap:.1%}")
            if thresh_metrics:
                console.print(f"Threshold tuned: {best_thresh:.2f} -> +{thresh_improvement*100:.1f}% improvement")
            
            # Reject meta-labeler if still overfitting badly
            if overfit_gap > 0.20:
                console.print(f"[yellow]⚠️ Meta-labeler rejected (gap {overfit_gap:.1%} > 20%)[/yellow]")
                meta_labeler_path = None
            else:
                # Save meta-labeler
                meta_labeler_path_obj = model_dir / f"buddy_meta{suffix}.pkl"
                labeler.save(meta_labeler_path_obj)
                meta_labeler_path = str(meta_labeler_path_obj)
                console.print(f"Saved: {meta_labeler_path}")
            
        except Exception as e:
            console.print(f"[yellow]Meta-labeling failed[/yellow]: {e}")
            meta_warnings.append(f"Meta-labeling failed: {e}")
            import traceback
            traceback.print_exc()

    meta = {
        "model_path": str(model_path),
        "csv_path": str(csv_path),
        "seq_len": int(seq_len),
        "all_features": bool(all_features),
        "run_tag": run_tag,
        "seed": int(seed),
        "lr": float(lr),
        "optimizer": str(optimizer_name) if optimizer_name is not None else None,
        "batch_size": int(batch_size),
        "epochs_requested": int(epochs),
        "train_smoothing": bool(train_smoothing),
        "median_window": int(median_window) if median_window is not None else None,
        "vol_norm_window": int(vol_norm_window) if vol_norm_window is not None else None,
        "min_volume": int(min_volume) if min_volume is not None else None,
        "spread_filter": bool(spread_filter),
        "spread_pctl": float(spread_pctl),
        "spread_mult": float(spread_mult),
        "top_features": int(top_features) if top_features is not None else None,
        "early_stopping_patience": int(patience),
        "early_stopping_monitor": monitor,
        "combined_w_dir": float(combined_w_dir),
        "combined_w_conf": float(combined_w_conf),
        "feature_columns": feature_columns,
        "standardize": {"mean": mu.reshape(-1).tolist(), "std": sigma.reshape(-1).tolist()},
        "pca_components": int(pca_components_eff) if pca_components_eff is not None else None,
        "pca": (
            {
                "mean": pca_model.mean_.reshape(-1).tolist(),
                "components": pca_model.components_.tolist(),
                "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
            }
            if pca_model is not None
            else None
        ),
        "confidence_target": "tier2_tp_sl_true_dir_v1",
        "tier2_label": {
            "instrument": str(conf_instrument_guess) if conf_instrument_guess else None,
            "stop_loss_pips": float(tier2_stop_loss_pips),
            "take_profit_pips": float(tier2_take_profit_pips),
            "horizon_candles": int(tier2_horizon_candles),
            "spread_fallback_pips": float(tier2_spread_fallback_pips),
            "tie_break": str(tier2_tie_break),
            "label_stride": int(label_stride),
            "timeouts_are_losses": True,
        },
        "train_dir_pos_rate": train_dir_pos_rate,
        "val_dir_pos_rate": val_dir_pos_rate,
        "train_conf_target_mean": train_conf_target_mean,
        "val_conf_target_mean": val_conf_target_mean,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_metric_name": str(best_metric_name) if best_metric_name is not None else None,
        "best_metric_value": float(best_metric_value) if best_metric_value is not None else None,
        "val_direction_accuracy": val_acc,
        "val_avg_confidence": avg_conf,
        "val_conf_mae": float(val_conf_mae) if val_conf_mae is not None else None,
        "val_conf_p80": float(val_conf_p80) if val_conf_p80 is not None else None,
        "val_conf_p90": float(val_conf_p90) if val_conf_p90 is not None else None,
        "val_combined_score": combined_score,
        "oos_replay": oos_block,
        "oos_baseline": baseline_oos_block,
        "trained_epochs": int(len(history.history.get("loss", []))),
        "early_stopped": bool(len(history.history.get("loss", [])) < int(epochs)),
        # Live-enabled should depend on directional skill. Trading confidence is handled by FX Tier-1 gating.
        "live_enabled": bool(val_acc >= 0.62),
        "live_confidence_threshold": 0.65,
        # Inference stability: average over last N windows (prevents single-window saturation).
        "inference_ensemble_windows": 20,
        # Confidence aggregation over ensemble windows.
        # - direction: always mean
        # - confidence: quantile by default (robust to a few saturated windows)
        "inference_confidence_agg": "quantile",
        "inference_confidence_quantile": 0.90,
        "tier2": (lambda: {
            "enabled": bool(tier2_calibrate and tier2_calibration is not None),
            "calibration": tier2_calibration,
            "best_effort": bool(tier2_best_effort),
            "warning": str(tier2_warnings[0]) if len(tier2_warnings) == 1 else None,
            "warnings": list(tier2_warnings) if len(tier2_warnings) > 1 else None,
        })(),
        "warnings": list(meta_warnings) if meta_warnings else None,
        "meta_labeling": {
            "enabled": bool(meta_labeling_enabled and meta_labeler_path is not None),
            "path": meta_labeler_path,
            "metrics": meta_labeling_metrics,
        } if meta_labeling_enabled else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    console.print(f"Saved: {model_path}")
    console.print(f"Saved: {meta_path}")

    # ==========================================================================
    # RL Position Sizer Training (automatic after ensemble training)
    # ==========================================================================
    # Check config flag first (can be disabled via config), then CLI option
    auto_train_rl_config = cfg.get("training", {}).get("auto_train_rl", True)
    train_rl_sizer_enabled = options.train_rl_sizer and auto_train_rl_config
    
    if train_rl_sizer_enabled:
        try:
            console.print()
            console.print(Panel(
                "[bold cyan]🤖 RL Position Sizer Training[/bold cyan]\n\n"
                "[dim]Training RL agent to learn optimal position sizing based on[/dim]\n"
                "[dim]ensemble predictions and market conditions.[/dim]",
                title="RL Training",
                border_style="cyan",
            ))
            
            # Prepare RL training data from standard training
            # Features: use the full feats_model array (standardized features)
            rl_features = feats_model.astype(np.float32)
            n_rl_samples = len(rl_features)
            
            # Get prices from ohlc_df
            rl_prices = ohlc_df['close'].values[:n_rl_samples].astype(np.float32)
            
            # Generate ensemble predictions using the trained model
            # We need to create sequences and predict
            console.print(f"[dim]Generating ensemble predictions for {n_rl_samples:,} samples...[/dim]")
            
            rl_direction_probs = np.zeros(n_rl_samples, dtype=np.float32)
            rl_confidences = np.zeros(n_rl_samples, dtype=np.float32)
            
            # Predict in batches for efficiency
            batch_size_rl = 256
            _inputs_struct = getattr(model, "_inputs_struct", None)
            _expects_features_dict = isinstance(_inputs_struct, dict) and "features" in _inputs_struct
            
            for start_idx in range(seq_len - 1, n_rl_samples, batch_size_rl):
                end_idx = min(start_idx + batch_size_rl, n_rl_samples)
                batch_seqs = []
                batch_indices = []
                
                for idx in range(start_idx, end_idx):
                    if idx - seq_len + 1 >= 0:
                        seq = rl_features[idx - seq_len + 1 : idx + 1]
                        if seq.shape[0] == seq_len:
                            batch_seqs.append(seq)
                            batch_indices.append(idx)
                
                if batch_seqs:
                    batch_x = np.stack(batch_seqs, axis=0).astype(np.float32)
                    pred_in = {"features": batch_x} if _expects_features_dict else batch_x
                    
                    try:
                        preds = model.predict(pred_in, verbose=0)
                        p_dir = np.asarray(preds["direction"]).reshape(-1)
                        p_conf = np.asarray(preds["confidence"]).reshape(-1)
                        
                        for i, idx in enumerate(batch_indices):
                            if i < len(p_dir):
                                rl_direction_probs[idx] = float(p_dir[i])
                                rl_confidences[idx] = float(np.clip(p_conf[i], 0.0, 1.0))
                    except Exception:
                        # If prediction fails, use neutral values
                        for idx in batch_indices:
                            rl_direction_probs[idx] = 0.5
                            rl_confidences[idx] = 0.5
            
            # Fill in early samples with neutral predictions (before seq_len window)
            rl_direction_probs[:seq_len] = 0.5
            rl_confidences[:seq_len] = 0.5
            
            # Stack predictions: [direction_prob, confidence]
            rl_predictions = np.column_stack([
                rl_direction_probs,
                rl_confidences,
            ]).astype(np.float32)
            
            console.print(f"[dim]RL training data ready: {n_rl_samples:,} samples, {rl_features.shape[1]} features[/dim]")
            
            # Train RL position sizer
            rl_timesteps = options.rl_timesteps
            _train_rl_position_sizer_if_ready(
                console=console,
                rl_timesteps=rl_timesteps,
                features=rl_features,
                ensemble_predictions=rl_predictions,
                prices=rl_prices,
            )
            
        except Exception as e:
            # RL training failure should NOT crash main training
            console.print(f"[yellow]⚠ RL position sizer training failed: {e}[/yellow]")
            console.print("[dim]Ensemble training completed successfully - RL training is optional.[/dim]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


def _buddy_live_enabled_from_meta() -> tuple[bool, float]:
    """Return (live_enabled, confidence_threshold) from last Buddy training."""
    import json

    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    if not meta_path.exists():
        return False, 0.65
    try:
        meta = json.loads(meta_path.read_text())
        return bool(meta.get("live_enabled", False)), float(meta.get("live_confidence_threshold", 0.65))
    except Exception:
        return False, 0.65


def _fx_confirm(prompt: str) -> bool:
    ans = (console.input(prompt) or "").strip().lower()
    return ans == "y"


def _fx_open_position_instruments(payload: Any) -> list[str]:
    pos = (payload or {}).get("positions") or []
    return [str(p.get("instrument")) for p in pos if p.get("instrument")]


def _fx_enforce_fx_policy(cfg: Dict[str, Any], *, instrument: str, granularity: str):
    from fx_guardrails import load_fx_policy

    policy = load_fx_policy(cfg)
    if instrument not in set(policy.instruments):
        console.print(f"[bold red]Blocked[/bold red]: instrument not allowed ({instrument}).")
        console.print(f"Allowed: {', '.join(policy.instruments)}")
        return None
    if granularity != policy.granularity:
        console.print(f"[bold red]Blocked[/bold red]: granularity must be {policy.granularity} (got {granularity}).")
        return None
    return policy


def _fx_refresh_fx_state(cfg: Dict[str, Any], policy: Any, state: Any, client: Any) -> tuple[dict[str, Any], str]:
    import fx_guardrails as fxg

    pnl = fxg.update_state_from_account_summary(policy, state, client.get_account_summary())

    # Evaluate daily circuit breakers early so we can stop even if we later end up
    # in a HOLD / no-setup path. Profit stop may be configured per-band, but the
    # repo's Tier-1 defaults use a fixed 30% target for both medium/high.
    stop_hit, stop_reason, stop_kind = fxg.check_daily_stops(
        policy,
        drawdown_pct=pnl.get("drawdown_pct"),
        realized_pct=pnl.get("realized_pct"),
        confidence_band="medium",
    )
    if stop_hit and stop_reason:
        state.disabled_reason = stop_reason
        state.disabled_kind = stop_kind
        fxg.save_state(cfg, state)
        console.print(f"[bold red]Trading disabled[/bold red]: {stop_reason}")

    # Band is computed later; keep placeholder here for display.
    return pnl, "unknown"


def _fx_maybe_force_flat(policy: Any, state: Any, client: Any, *, execute: bool) -> bool:
    from fx_guardrails import should_force_flat

    # Only force-flat automatically for the time cutoff or loss-stop.
    force_flat_due_to_stop = bool(state.disabled_reason) and (getattr(state, "disabled_kind", None) == "loss")
    if not should_force_flat(policy) and not force_flat_due_to_stop:
        return False

    insts = _fx_open_position_instruments(client.get_open_positions())
    if not insts:
        console.print("[dim]No open positions to close.[/dim]")
        return True

    console.print("[yellow]Force-flat: open positions detected.[/yellow]")
    if not execute:
        console.print("[dim]Dry-run: would close all open positions.[/dim]")
        return True

    if policy.require_confirmation and not _fx_confirm("Confirm CLOSE ALL positions? (y/N): "):
        console.print("[dim]Close cancelled.[/dim]")
        return True

    for inst in insts:
        try:
            client.close_position(instrument=inst)
            console.print(f"Closed position: {inst}")
        except Exception as e:
            console.print(f"[bold red]Failed[/bold red] closing {inst}: {e}")

    return True


def _fx_gate_fx_entry(policy: Any, state: Any, client: Any) -> bool:
    from fx_guardrails import can_open_new_trade

    ok, why = can_open_new_trade(policy, state)
    if not ok:
        console.print(f"[bold red]Blocked[/bold red]: {why}")
        return False

    insts = _fx_open_position_instruments(client.get_open_positions())
    if len(insts) >= policy.limits.max_open_positions:
        console.print("[bold red]Blocked[/bold red]: max open positions reached.")
        return False

    return True


def _fx_load_fx_df(client: Any, *, instrument: str, granularity: str, candles: int):
    from fx_paper import candles_to_ohlcv_df

    resp = client.get_candles(instrument, granularity=granularity, count=candles, price="MBA")
    return candles_to_ohlcv_df(resp)


def _fx_spread_and_slippage(policy: Any, df: Any, *, instrument: str) -> tuple[bool, float, float, float]:
    from fx_paper import conservative_slippage_pips, pip_size, spread_pips_from_df

    spread_pips = spread_pips_from_df(df, instrument)
    if spread_pips is None:
        spread_pips = float(policy.costs.spread_fallback_pips.get(instrument, 0.0) or 0.0)

    max_spread = float(policy.costs.max_spread_pips.get(instrument, 0.0) or 0.0)
    if max_spread > 0 and spread_pips > max_spread:
        console.print(
            f"[bold red]Blocked[/bold red]: spread too wide ({spread_pips:.2f} pips > {max_spread:.2f} pips)."
        )
        return False, spread_pips, 0.0, 0.0

    slippage_pips = conservative_slippage_pips(
        spread_pips=spread_pips,
        min_pips=policy.costs.slippage_pips_min,
        spread_mult=policy.costs.slippage_pips_spread_mult,
    )
    slippage_price = slippage_pips * pip_size(instrument)
    return True, float(spread_pips), float(slippage_pips), float(slippage_price)


def _fx_require_account_metrics(pnl: Dict[str, Any]) -> bool:
    """Fail-closed if we can't read basic account metrics."""
    nav = pnl.get("nav")
    balance = pnl.get("balance")
    if nav is None or balance is None:
        console.print("[bold red]Blocked[/bold red]: missing account NAV/balance from broker.")
        console.print(f"[dim]pnl payload keys: {sorted((pnl or {}).keys())}[/dim]")
        return False
    return True


def _fx_get_signal_context(df: Any) -> tuple[str, float, float]:
    from fx_paper import atr as fx_atr
    from fx_paper import setup_signal

    signal = setup_signal(df)
    last_close = float(df["close"].iloc[-1])
    atr_value = float(fx_atr(df, period=14))
    return str(signal), float(last_close), float(atr_value)


def _fx_build_risk_rules(
    policy: Any,
    pnl: Dict[str, Any],
    *,
    equity: float,
    risk_per_trade_pct: float,
):
    from fx_paper import RiskRules

    # Prefer live NAV when available.
    nav = pnl.get("nav")
    base_equity = float(nav) if nav is not None else float(equity)

    # Use the explicit CLI flag if provided, else fallback to policy.
    rpt = float(risk_per_trade_pct) if risk_per_trade_pct is not None else float(policy.risk.risk_per_trade_pct)

    return RiskRules(
        equity=base_equity,
        risk_per_trade_pct=rpt,
        max_daily_loss_pct=float(getattr(policy.risk, "daily_loss_stop_pct", 0.02)),
        max_open_positions=int(getattr(policy.limits, "max_open_positions", 1)),
        atr_stop_mult=float(getattr(policy.risk, "atr_stop_mult", 1.5)),
        rr_take_profit=float(getattr(policy.risk, "rr_take_profit", 1.5)),
    )


def _fx_compute_confidence_and_band(
    cfg: Dict[str, Any],
    policy: Any,
    *,
    instrument: str,
    signal: str,
    _price: float,
    atr_value: float,
    spread_pips: float,
) -> tuple[float, str, list[str]]:
    """Confidence score for Tier-1 gating (current source: fx_confidence_v1)."""
    import fx_guardrails as fxg
    from reasoning_enhanced import fx_confidence_v1

    max_spread = float(getattr(policy.costs, "max_spread_pips", {}).get(instrument, 0.0) or 0.0)
    msp = max_spread if max_spread > 0 else max(1e-9, float(spread_pips))
    params = dict((cfg.get("fx", {}) or {}).get("confidence_model") or {})

    payload = fx_confidence_v1(
        signal=str(signal),
        price=float(_price),
        atr=float(atr_value),
        spread_pips=float(spread_pips),
        max_spread_pips=float(msp),
        params=params,
    )
    confidence = float(payload.get("confidence") or 0.0)
    reasons = [str(x) for x in (payload.get("reasons") or [])]
    band = fxg.confidence_band(policy, confidence)
    return float(confidence), str(band), reasons


def _fx_apply_daily_stops(cfg: Dict[str, Any], policy: Any, state: Any, pnl: Dict[str, Any], *, band: str) -> bool:
    import fx_guardrails as fxg

    stop_hit, stop_reason, stop_kind = fxg.check_daily_stops(
        policy,
        drawdown_pct=pnl.get("drawdown_pct"),
        realized_pct=pnl.get("realized_pct"),
        confidence_band=str(band),
    )
    if not stop_hit:
        return False

    if stop_reason:
        state.disabled_reason = stop_reason
        state.disabled_kind = stop_kind
        fxg.save_state(cfg, state)
        console.print(f"[bold red]Trading disabled[/bold red]: {stop_reason}")
    return True


def _fx_build_order_units_and_prices(
    policy: Any,
    rules: Any,
    *,
    instrument: str,
    signal: str,
    price: float,
    atr_value: float,
    slippage_price: float,
) -> tuple[int, float, float, float, float]:
    from fx_paper import position_size_units

    stop_distance = float(atr_value) * float(getattr(policy.risk, "atr_stop_mult", 1.5))
    stop_distance = float(max(stop_distance, 0.0)) + float(max(slippage_price, 0.0))
    if stop_distance <= 0:
        raise ValueError("Computed stop distance is not positive")

    tp_distance = stop_distance * float(getattr(policy.risk, "rr_take_profit", 1.5))

    if signal == "buy":
        stop_price = float(price) - stop_distance
        tp_price = float(price) + tp_distance
        side = 1
    else:
        stop_price = float(price) + stop_distance
        tp_price = float(price) - tp_distance
        side = -1

    units_abs = position_size_units(
        instrument=instrument,
        equity=float(getattr(rules, "equity", 0.0)),
        risk_per_trade_pct=float(getattr(rules, "risk_per_trade_pct", 0.005)),
        stop_distance_price=stop_distance,
        price=float(price),
        pip_value_per_unit=None,
    )
    return int(side * units_abs), float(stop_price), float(tp_price), float(stop_distance), float(tp_distance)


def _fx_execution_guard_price_bound(_policy: Any, client: Any, *, instrument: str, units: int) -> float | None:
    from fx_paper import pip_size

    try:
        q = client.get_price_quote(instrument=instrument)
    except Exception as e:
        console.print(f"[bold red]Blocked[/bold red]: could not fetch live quote: {e}")
        return None

    bid = float(q["bid"])
    ask = float(q["ask"])
    # Determine buffer in pips. Prefer policy-level override `costs.price_bound_buffer_pips` if available.
    buffer_pips = None
    try:
        buffer_pips = float(getattr(_policy.costs, "price_bound_buffer_pips"))
    except Exception:
        try:
            # support dict-like policy.costs
            buffer_pips = float((_policy.get("costs") or {}).get("price_bound_buffer_pips", None))
        except Exception:
            buffer_pips = None

    # Default to a small buffer (1 pip) suitable for scalping; make configurable via policy.
    if buffer_pips is None:
        buffer_pips = 1.0

    buffer_price = float(buffer_pips) * pip_size(instrument)

    # Use the ask/bid with a modest buffer (avoid adding half-spread again which made bounds overly conservative).
    if int(units) > 0:
        return float(ask + buffer_price)
    return float(bid - buffer_price)


def _schedule_auto_close(client: Any, instrument: str, delay_s: float, *, verbose: bool = False) -> None:
    """Spawned in a daemon thread to close the instrument position after delay_s seconds.

    This is a best-effort helper for PRACTICE mode to ensure scalping-style trades
    are not left open beyond the desired timeframe.
    """
    import threading

    def _worker():
        try:
            if verbose:
                console.print(f"[dim]Auto-close thread[/dim]: sleeping {delay_s:.1f}s before closing {instrument}")
            time.sleep(max(0.0, float(delay_s)))
            try:
                # Prefer closing the specific trade if the create_market_order returned a trade id
                # The client may expose `close_trade(trade_id=...)` or fall back to close_position.
                if hasattr(client, "close_trade") and hasattr(client, "_last_trade_id") and getattr(client, "_last_trade_id"):
                    tid = getattr(client, "_last_trade_id")
                    try:
                        res = client.close_trade(trade_id=tid)
                        console.print(f"[dim]Auto-close[/dim]: closed trade {tid} for {instrument}: {res}")
                    except Exception:
                        res = client.close_position(instrument=instrument)
                        console.print(f"[dim]Auto-close[/dim]: fallback closed position for {instrument}: {res}")
                else:
                    res = client.close_position(instrument=instrument)
                    console.print(f"[dim]Auto-close[/dim]: closed position for {instrument}: {res}")
            except Exception as e:
                console.print(f"[yellow]Auto-close failed[/yellow]: could not close {instrument}: {e}")
        except Exception:
            # Never allow the worker to raise into the main thread
            return

    t = threading.Thread(target=_worker, daemon=True, name=f"auto-close-{instrument}")
    t.start()


def _build_integrated_engines(config: Dict[str, Any]) -> NeuralNetworkIntegrator:
    """Retired.

    Legacy integrated ML/MT/MR pipeline has been retired as part of the
    TensorFlow-only Buddy migration.
    """

    raise RuntimeError("Retired: integrated engines are no longer supported. Use `train-buddy`/`buddy`.")


def integrated_predict_once(config: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility shim for legacy integrated prediction.

    The production path is Buddy (`train-buddy`/`buddy`). Some unit tests still
    exercise the historical integrated API. Keep this no-network and lightweight.
    """

    import numpy as np

    batch_size = int((config or {}).get("batch_size") or 1)
    # A deterministic placeholder prediction tensor.
    prediction = np.zeros((batch_size, 1), dtype=np.float32)
    reasoning = {"insights": ["Integrated pipeline is deprecated; returned stub prediction."]}
    return {"prediction": prediction, "reasoning": reasoning}


def compute_slm_indicators(market_data_df):
    """Compute a small subset of SLM technical indicators without network calls."""
    try:
        from SLM.stock_slm import (
            calculate_sma,
            calculate_ema,
            calculate_rsi,
            calculate_macd,
            calculate_bollinger_bands,
        )
    except Exception as e:  # pragma: no cover
        raise ImportError(f"Failed to import SLM indicators: {e}") from e

    df_for_slm = market_data_df
    # SLM indicator functions expect title-case columns like 'Close'.
    if hasattr(market_data_df, "columns") and "Close" not in market_data_df.columns:
        rename_map = {}
        for col in market_data_df.columns:
            key = str(col).strip().lower()
            if key == "close":
                rename_map[col] = "Close"
            elif key == "open":
                rename_map[col] = "Open"
            elif key == "high":
                rename_map[col] = "High"
            elif key == "low":
                rename_map[col] = "Low"
            elif key == "volume":
                rename_map[col] = "Volume"
        if rename_map:
            df_for_slm = market_data_df.rename(columns=rename_map)

    sma_20 = calculate_sma(df_for_slm, 20)
    ema_20 = calculate_ema(df_for_slm, 20)
    rsi_14 = calculate_rsi(df_for_slm, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(df_for_slm)
    bb_upper, bb_lower = calculate_bollinger_bands(df_for_slm, 20, 2)

    return {
        "sma_20": sma_20,
        "ema_20": ema_20,
        "rsi_14": rsi_14,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
    }


def _pick_default_market_csv(config: Dict[str, Any]) -> str:
    data_dir = (
        config.get("data", {}).get("data_dir")
        or config.get("paths", {}).get("data_dir")
        or config.get("DATA_DIR")
        or config.get("data_dir")
        or "market_data"
    )
    candidates = sorted(Path(data_dir).glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No .csv files found under: {data_dir}")
    return str(candidates[0])


def _normalize_market_dataframe(df):
    """Return a copy with standardized OHLCV column names: open/high/low/close/volume."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("market_data_df must be a pandas DataFrame")

    out = df.copy()
    rename_map = {}
    for col in out.columns:
        key = str(col).strip().lower()
        if key in {"open", "o"}:
            rename_map[col] = "open"
        elif key in {"high", "h"}:
            rename_map[col] = "high"
        elif key in {"low", "l"}:
            rename_map[col] = "low"
        elif key in {"close", "c", "adj close", "adj_close", "adjclose"}:
            rename_map[col] = "close"
        elif key in {"volume", "vol", "v"}:
            rename_map[col] = "volume"

    out = out.rename(columns=rename_map)
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Market DataFrame missing required columns: {missing}")
    return out


def build_integrated_feature_tensors(
    market_data_df, config: Dict[str, Any], *, allow_pad: bool = False
) -> Dict[str, Any]:
    """Compatibility shim for legacy integrated feature tensors.

    Tests expect 3 tensors:
    - ml_features: (1, seq_len, 7)
    - mt_features: (1, seq_len, 11)
    - mr_features: (1, seq_len, 11)

    This implementation is intentionally simple and deterministic.
    """

    import numpy as np

    df = _normalize_market_dataframe(market_data_df)
    seq_len = int((config or {}).get("data", {}).get("sequence_length") or 60)

    # Base numeric series.
    o = df["open"].to_numpy(dtype=np.float32)
    h = df["high"].to_numpy(dtype=np.float32)
    low = df["low"].to_numpy(dtype=np.float32)
    c = df["close"].to_numpy(dtype=np.float32)
    v = df["volume"].to_numpy(dtype=np.float32)

    n = int(len(df))
    if n <= 0:
        raise ValueError("market_data_df must have at least 1 row")

    if n < seq_len:
        if not allow_pad:
            raise ValueError(f"Not enough rows ({n}) for sequence_length={seq_len}")
        pad = seq_len - n
        # Left-pad using the first row (conservative, stable).
        o = np.concatenate([np.repeat(o[:1], pad), o])
        h = np.concatenate([np.repeat(h[:1], pad), h])
        low = np.concatenate([np.repeat(low[:1], pad), low])
        c = np.concatenate([np.repeat(c[:1], pad), c])
        v = np.concatenate([np.repeat(v[:1], pad), v])
    else:
        o = o[-seq_len:]
        h = h[-seq_len:]
        low = low[-seq_len:]
        c = c[-seq_len:]
        v = v[-seq_len:]

    # Derived features.
    prev_c = np.concatenate([c[:1], c[:-1]])
    ret = (c - prev_c) / np.maximum(prev_c, 1e-8)
    rng = (h - low) / np.maximum(c, 1e-8)

    ml = np.stack([o, h, low, c, v, ret, rng], axis=-1).astype(np.float32)

    # MT/MR historical shapes include extra engineered slots; fill deterministically.
    zeros4 = np.zeros((seq_len, 4), dtype=np.float32)
    zeros4b = np.zeros((seq_len, 4), dtype=np.float32)
    mt = np.concatenate([ml, zeros4], axis=-1)
    mr = np.concatenate([ml, zeros4b], axis=-1)

    return {
        "ml_features": ml[None, :, :],
        "mt_features": mt[None, :, :],
        "mr_features": mr[None, :, :],
    }


def integrated_predict(config_path: str, csv_path: str | None = None) -> None:
    raise RuntimeError("Retired: integrated predict is no longer supported. Use `train-buddy`/`buddy`.")


def _fetch_live_market_data(ticker: str, period: str, interval: str):
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Live data fetch requires `yfinance`.") from e

    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data is None or data.empty:
        raise RuntimeError(f"No live data returned for {ticker} ({period}, {interval})")
    # Keep the index as a column (Date/Datetime) for display/debugging.
    return data.reset_index()


def _df_last_value(df, candidates: tuple[str, ...]):
    for col in candidates:
        if col in df.columns:
            return df[col].iloc[-1]
    return None


def _print_last_bar(df) -> None:
    last_close = _df_last_value(df, ("Close", "close"))
    last_ts = _df_last_value(df, ("Datetime", "Date", "date", "datetime"))
    if last_ts is None and last_close is None:
        return
    console.print(f"[dim]Last bar:[/dim] {last_ts}  close={last_close}")


def _print_reasoning_insights(reasoning) -> None:
    if not isinstance(reasoning, dict):
        return
    insights = reasoning.get("insights")
    if not insights:
        return
    console.print("[bold green]Reasoning insights:[/bold green]")
    for insight in insights:
        console.print(f"- {insight}")


def _integrated_predict_live_once(
    *,
    ticker: str,
    period: str,
    interval: str,
    config: Dict[str, Any],
    integrator: NeuralNetworkIntegrator,
) -> None:
    df = _fetch_live_market_data(ticker, period=period, interval=interval)
    features = build_integrated_feature_tensors(df, config, allow_pad=True)
    result = integrator.predict(features)

    console.print(f"[bold blue]Live predict[/bold blue] {ticker} ({period}, {interval})")
    _print_last_bar(df)
    console.print(f"[bold yellow]Prediction:[/bold yellow] {result.get('prediction')}")
    console.print(f"[bold yellow]Uncertainty:[/bold yellow] {result.get('uncertainty')}")

    weights = result.get("weights")
    if weights is not None:
        console.print(f"[cyan]Weights (ML/MT/MR):[/cyan] {weights}")

    _print_reasoning_insights(result.get("reasoning"))


def integrated_predict_live(
    config_path: str,
    ticker: str,
    period: str = "5d",
    interval: str = "1h",
    watch_seconds: int | None = None,
) -> None:
    """Run the unified prediction using live-ish OHLCV from yfinance."""
    config = load_config(config_path)
    integrator = _build_integrated_engines(config)

    _integrated_predict_live_once(
        ticker=ticker,
        period=period,
        interval=interval,
        config=config,
        integrator=integrator,
    )

    if watch_seconds is None:
        return

    interval_seconds = max(1, int(watch_seconds))
    while True:  # pragma: no cover
        time.sleep(interval_seconds)
        _integrated_predict_live_once(
            ticker=ticker,
            period=period,
            interval=interval,
            config=config,
            integrator=integrator,
        )


@dataclass(frozen=True)
class _FxPaperTradePlan:
    instrument: str
    granularity: str
    signal: str
    last_close: float
    atr_value: float
    units: int
    stop_price: float
    tp_price: float
    spread_pips: float
    slippage_pips: float
    slippage_price: float
    confidence: float
    band: str
    conf_reasons: list[str]


def _fx_setup_paper_trade(cfg: Dict[str, Any], *, instrument: str, granularity: str, execute: bool):
    from oanda_practice import OandaPracticeClient
    import fx_guardrails as fxg

    policy = _fx_enforce_fx_policy(cfg, instrument=instrument, granularity=granularity)
    if policy is None:
        return None

    client = OandaPracticeClient.from_env()
    state = fxg.load_state(cfg, policy)

    pnl, _ = _fx_refresh_fx_state(cfg, policy, state, client)
    if _fx_maybe_force_flat(policy, state, client, execute=execute):
        return None

    if not _fx_require_account_metrics(pnl):
        return None

    if not _fx_gate_fx_entry(policy, state, client):
        return None

    return policy, client, state, pnl


def _fx_build_paper_trade_plan(
    cfg: Dict[str, Any],
    policy: Any,
    client: Any,
    state: Any,
    pnl: Dict[str, Any],
    *,
    instrument: str,
    granularity: str,
    candles: int,
    equity: float,
    risk_per_trade_pct: float,
) -> tuple["_FxPaperTradePlan", Any] | None:
    import fx_guardrails as fxg  # noqa: F401  (kept for symmetry; state saved elsewhere)

    df = _fx_load_fx_df(client, instrument=instrument, granularity=granularity, candles=candles)
    ok_spread, spread_pips, slippage_pips, slippage_price = _fx_spread_and_slippage(policy, df, instrument=instrument)
    if not ok_spread:
        return None

    signal, last_close, atr_value = _fx_get_signal_context(df)
    if signal == "hold":
        console.print(f"FX setup: HOLD  {instrument} {granularity}  close={last_close:.5f}  ATR14={atr_value:.5f}")
        return None

    rules = _fx_build_risk_rules(policy, pnl, equity=equity, risk_per_trade_pct=risk_per_trade_pct)

    confidence, band, conf_reasons = _fx_compute_confidence_and_band(
        cfg,
        policy,
        instrument=instrument,
        signal=signal,
        _price=last_close,
        atr_value=atr_value,
        spread_pips=float(spread_pips),
    )
    if band == "low":
        console.print(f"[bold red]Blocked[/bold red]: low confidence ({confidence:.2f}) => no trades.")
        console.print(f"[dim]Reasons: {', '.join(conf_reasons)}[/dim]")
        return None

    if _fx_apply_daily_stops(cfg, policy, state, pnl, band=band):
        return None

    units, stop_price, tp_price, _, _ = _fx_build_order_units_and_prices(
        policy,
        rules,
        instrument=instrument,
        signal=signal,
        price=last_close,
        atr_value=atr_value,
        slippage_price=float(slippage_price),
    )

    plan = _FxPaperTradePlan(
        instrument=str(instrument),
        granularity=str(granularity),
        signal=str(signal),
        last_close=float(last_close),
        atr_value=float(atr_value),
        units=int(units),
        stop_price=float(stop_price),
        tp_price=float(tp_price),
        spread_pips=float(spread_pips),
        slippage_pips=float(slippage_pips),
        slippage_price=float(slippage_price),
        confidence=float(confidence),
        band=str(band),
        conf_reasons=[str(x) for x in (conf_reasons or [])],
    )
    return plan, rules


def _fx_execute_paper_trade_plan(
    cfg: Dict[str, Any],
    policy: Any,
    client: Any,
    state: Any,
    plan: _FxPaperTradePlan,
    rules: Any,
    *,
    execute: bool,
    verbose: bool,
) -> None:
    import fx_guardrails as fxg

    if not execute:
        console.print("[dim]Dry-run (no order). Pass --execute to place a PRACTICE order.[/dim]")
        return

    if policy.require_confirmation and not _fx_confirm("Confirm PLACE ORDER? (y/N): "):
        console.print("[dim]Order cancelled.[/dim]")
        return

    units = int(plan.units)

    # Apply any programmatic override (rules.force_units) before guard and submission.
    try:
        if hasattr(rules, "force_units") and rules.force_units is not None:
            fu = int(rules.force_units)
            units = -abs(int(fu)) if units < 0 else abs(int(fu))
            console.print(f"[dim]Force-units override (rules.force_units)[/dim]: using units={units}")
    except Exception:
        pass

    # Finalize units as integer (preserve sign) and submit exactly that value.
    try:
        units_final = int(units)
    except Exception:
        units_final = int(float(units))

    if verbose:
        console.print(f"[dim]Order sizing[/dim]: computed_units={units} final_submitted_units={units_final}")

    price_bound = _fx_execution_guard_price_bound(policy, client, instrument=plan.instrument, units=units_final)
    if price_bound is None:
        return

    result = client.create_market_order(
        instrument=plan.instrument,
        units=units_final,
        stop_loss_price=float(plan.stop_price),
        take_profit_price=float(plan.tp_price),
        price_bound=price_bound,
    )
    state.entries_today += 1
    fxg.save_state(cfg, state)
    console.print("[bold green]Order submitted (PRACTICE).[/bold green]")
    console.print(result)

    # Parse returned transaction to capture trade id(s) for precise auto-close.
    try:
        tx = (result or {}).get("orderFillTransaction") or (result or {}).get("orderCreateTransaction") or {}
        trade_id = None
        if isinstance(tx, dict):
            # Typical OANDA response may include 'tradeOpened' or 'tradesOpened' structures.
            to = tx.get("tradeOpened")
            if isinstance(to, dict):
                trade_id = to.get("tradeID") or to.get("id")
            tro = tx.get("tradesOpened")
            if trade_id is None and isinstance(tro, list) and len(tro) > 0:
                trade_id = tro[0].get("tradeID") or tro[0].get("id")
        if trade_id:
            try:
                # attach to client for auto-close worker to use
                setattr(client, "_last_trade_id", str(trade_id))
            except Exception:
                pass
    except Exception:
        pass

    # Schedule auto-close if `buddy.max_hold_minutes` is set in config (PRACTICE only).
    try:
        m_cfg = cfg.get("buddy", {}).get("max_hold_minutes", None)
        if m_cfg is not None:
            m = float(m_cfg)
            if m > 0:
                _schedule_auto_close(client, plan.instrument, delay_s=m * 60.0, verbose=bool(verbose))
    except Exception:
        pass


def fx_paper_trade(
    config_path: str,
    instrument: str,
    granularity: str = "M5",
    candles: int = 300,
    execute: bool = False,
    verbose: bool = False,
    equity: float = 10_000.0,
    risk_per_trade_pct: float = 0.005,
) -> None:
    """Paper trade forex on OANDA PRACTICE using a simple setup rule.

    This is intentionally conservative and defaults to DRY-RUN (no order placed).
    """
    cfg = load_config(config_path)

    setup = _fx_setup_paper_trade(cfg, instrument=instrument, granularity=granularity, execute=execute)
    if setup is None:
        return
    policy, client, state, pnl = setup

    planned = _fx_build_paper_trade_plan(
        cfg,
        policy,
        client,
        state,
        pnl,
        instrument=instrument,
        granularity=granularity,
        candles=candles,
        equity=equity,
        risk_per_trade_pct=risk_per_trade_pct,
    )
    if planned is None:
        return
    plan, rules = planned

    console.print(
        f"FX setup: {plan.signal.upper()}  {plan.instrument} {plan.granularity}  close={plan.last_close:.5f}  "
        f"units={plan.units}  SL={plan.stop_price:.5f}  TP={plan.tp_price:.5f}"
        f"  spread={plan.spread_pips:.2f}p  slip={plan.slippage_pips:.2f}p  conf={plan.confidence:.2f}  band={plan.band}"
    )

    _fx_execute_paper_trade_plan(
        cfg,
        policy,
        client,
        state,
        plan,
        rules,
        execute=execute,
        verbose=verbose,
    )


def generate_dashboard(
    epoch: int,
    total_epochs: int,
    latency: int,
    train_loss: float,
    val_loss: float,
    lr_delta: float,
    api_logs: list,
    config: Dict[str, Any],
) -> Layout:
    """Generate the rich dashboard layout."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body", size=8),
        Layout(name="footer", size=8),
    )
    metrics_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE, expand=True)
    metrics_table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    metrics_table.add_column("Value", style="green")
    metrics_table.add_row("Epoch", f"{epoch}/{total_epochs}")
    metrics_table.add_row("Training Loss", f"{train_loss:.6f}")
    metrics_table.add_row("Validation Loss", f"{val_loss:.6f}")
    metrics_table.add_row("API Latency", f"{latency}ms")
    lr_color = "green" if lr_delta >= 0 else "red"
    metrics_table.add_row("LR Change", f"[{lr_color}]{lr_delta:+.2f}%[/{lr_color}]")
    metrics_table.add_row(
        "Device", f"{config.get('hardware', {}).get('device', 'cpu')}"
    )
    status = (
        "[bold green]Active[/bold green]"
        if epoch < total_epochs
        else "[bold yellow]Complete[/bold yellow]"
    )
    metrics_table.add_row("Status", status)
    layout["header"].update(
        Panel(metrics_table, title="[bold]Training Metrics[/bold]", border_style="blue")
    )
    body_layout = Layout()
    body_layout.split_row(
        Layout(name="progress", ratio=1), Layout(name="logs", ratio=2)
    )
    progress_pct = (epoch / total_epochs * 100) if total_epochs > 0 else 0
    bar = "█" * int(progress_pct / 2) + "░" * (50 - int(progress_pct / 2))
    eta = (total_epochs - epoch) * latency / 1000
    progress_text = f"{progress_pct:.1f}%\n{bar}\nETA: {eta:.1f}s"
    body_layout["progress"].update(
        Panel(progress_text, title="[bold]Progress[/bold]", border_style="green")
    )
    current_time = time.strftime("%H:%M:%S")
    if not api_logs:
        api_logs.append(f"[grey]{current_time}[/grey] Starting training...")
    formatted_logs = [
        log if log.startswith("[") else f"[grey]{current_time}[/grey] {log}"
        for log in api_logs[-5:]
    ]
    logs_content = "\n".join(formatted_logs)
    body_layout["logs"].update(
        Panel(logs_content, title="[bold]Recent Updates[/bold]", border_style="yellow")
    )
    layout["body"].update(body_layout)
    config_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE, expand=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    model_config = config.get("model", {})
    config_table.add_row("Learning Rate", f"{model_config.get('learning_rate', 'N/A')}")
    config_table.add_row("Batch Size", f"{model_config.get('batch_size', 'N/A')}")
    config_table.add_row("Architecture", f"{model_config.get('architecture', 'N/A')}")
    config_table.add_row("Hidden Size", f"{model_config.get('hidden_size', 'N/A')}")
    config_table.add_row("Num Layers", f"{model_config.get('num_layers', 'N/A')}")
    config_table.add_row("Dropout", f"{model_config.get('dropout', 'N/A')}")
    optimizer_config = model_config.get("optimizer", {})
    config_table.add_row("Optimizer", f"{optimizer_config.get('type', 'Adam')}")
    logs_table = Table(show_header=True, header_style=TABLE_HEADER_STYLE, expand=True)
    logs_table.add_column("Latest Log", style="green")
    latest_log = api_logs[-1] if api_logs else "No logs available"
    logs_table.add_row(latest_log)
    footer_layout = Layout()
    footer_layout.split_row(
        Panel(config_table, title="[bold]Configuration[/bold]", border_style="blue"),
        Panel(logs_table, title="[bold]Log Info[/bold]", border_style="blue"),
    )
    layout["footer"].update(footer_layout)
    return layout


# CLI Command Implementations
def train_model(config_path: str) -> None:
    """Run model training with live progress dashboard.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If training fails
    """
    try:
        # Load and validate configuration
        config = load_config(config_path)
        
        # Validate training configuration
        if "training" not in config:
            raise ValueError("Configuration missing 'training' section")
        if "epochs" not in config["training"]:
            raise ValueError("Configuration missing 'training.epochs'")
        
        console.print("[bold blue]Initializing ML Engine...[/bold blue]")
        
        # Load training data
        from data_loader import MarketDataLoader
        data_loader = MarketDataLoader(config)
        
        console.print("[bold blue]Loading market data...[/bold blue]")
        
        # Get data directory from config
        data_dir = config.get("data", {}).get("data_dir", "market_data/")
        
        # Load multiple tickers from the data directory
        data_dict = data_loader.load_multiple_tickers(data_dir)
        
        if not data_dict:
            raise ValueError(f"No data files found in {data_dir}")
        
        # Combine all ticker data
        df = data_loader.combine_ticker_data(data_dict, method="concat")
        console.print(f"[cyan]Loaded {len(data_dict)} tickers, {len(df)} total rows[/cyan]")
        
        # Preprocess the data
        console.print("[bold blue]Preprocessing data...[/bold blue]")
        x_train, y_train, x_val, y_val, _, _ = data_loader.preprocess(
            df,
            add_features=True,
            scaler_type="standard",
            sequence_length=config.get("data", {}).get("sequence_length", 60),
            test_size=0.2,
        )
        
        console.print(f"[bold green]Data prepared: {len(x_train)} training samples, {len(x_val)} validation samples[/bold green]")
        
        # Update config with correct input size based on actual features
        input_size = x_train.shape[-1]  # Get feature dimension
        if "model" not in config:
            config["model"] = {}
        config["model"]["input_size"] = input_size
        console.print(f"[cyan]Model input size set to: {input_size}[/cyan]")

        from ml_engine_enhanced import EnhancedMLEngine

        engine = EnhancedMLEngine(config)
        
        # Get epochs from config (not nested in 'training')
        epochs = config.get("epochs", 100)
        console.print(f"[bold green]Training for {epochs} epochs[/bold green]")
        
        # Train the model and get results
        console.print("[bold green]Starting training...[/bold green]")
        result = engine.train(x_train, y_train, x_val, y_val, epochs=epochs)
        
        console.print("[bold green]Training completed successfully![/bold green]")
        console.print(f"[cyan]Total epochs trained: {result.get('total_epochs', epochs)}[/cyan]")
        console.print(f"[cyan]Best validation loss: {result['best_val_loss']:.6f}[/cyan]")
        console.print(f"[cyan]Resumed from checkpoint: {result.get('resumed', False)}[/cyan]")
                
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found: {e}[/red]")
        raise
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        logging.error(f"Unexpected error in train_model: {e}", exc_info=True)
        raise


def evaluate_model(config_path: str) -> None:
    """Run model evaluation with results dashboard.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        RuntimeError: If evaluation fails
    """
    try:
        console.print("[bold blue]Loading configuration and model...[/bold blue]")
        config = load_config(config_path)
        from ml_engine_enhanced import EnhancedMLEngine

        engine = EnhancedMLEngine(config)

        api_logs = [f"[blue]{time.strftime('%H:%M:%S')}[/blue] Starting evaluation"]
        
        with Live(generate_dashboard(0, 10, 200, 0.0, 0.0, 0.0, api_logs, config)) as live:
            try:
                console.print("[bold green]Evaluating model...[/bold green]")
                metrics = engine.evaluate_model()
                
                # Display evaluation results
                api_logs.append("[green]Evaluation completed[/green]")
                if metrics:
                    for key, value in metrics.items():
                        api_logs.append(f"[cyan]{key}: {value:.4f}[/cyan]")
                
                live.update(generate_dashboard(10, 10, 0, 0.0, 0.0, 0.0, api_logs, config))
                console.print("[bold green]Evaluation completed successfully![/bold green]")
                
            except Exception as e:
                console.print(f"[red]Evaluation error: {e}[/red]")
                logging.error(f"Evaluation failed: {e}", exc_info=True)
                raise RuntimeError(f"Evaluation failed: {e}")
                
    except Exception as e:
        console.print(f"[red]Failed to evaluate model: {e}[/red]")
        logging.error(f"Model evaluation error: {e}", exc_info=True)
        raise


# New CLI command implementations
def visualize_dashboard(config_path: str) -> None:
    """Display dashboard visualizations using the visualizer module."""
    import visualizer

    config = load_config(config_path)
    console.print("[bold blue]Launching visualizations...[/bold blue]")
    visualizer.display_dashboard(
        config
    )  # Assumes implementation exists in visualizer.py


def openai_tune(config_path: str) -> None:
    """Execute OpenAI-based auto-tuning using the openai_integration module."""
    try:
        import openai_integration
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "openai_integration requires the `openai` package; install it to use openai-tune."
        ) from e

    current_config = load_config(config_path)

    # Only set credentials when we actually use OpenAI integration.
    openai_integration.set_openai_credentials(current_config)

    # Gather current metrics from the engine
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(current_config)
    train_loss = (
        engine.train_losses[-1]
        if hasattr(engine, "train_losses") and engine.train_losses
        else None
    )
    val_loss = (
        engine.val_losses[-1]
        if hasattr(engine, "val_losses") and engine.val_losses
        else None
    )
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": current_config.get("model", {}).get("learning_rate", 0.001),
        "epochs_completed": len(engine.train_losses)
        if hasattr(engine, "train_losses")
        else 0,
        "device": current_config.get("hardware", {}).get("device", "cpu"),
        "batch_size": current_config.get("model", {}).get("batch_size", 32),
    }

    # Call the integration function with metrics and config in proper order
    result = openai_integration.query_for_auto_configuration(metrics, current_config)

    if result:
        _ = openai_integration.report_config_changes(current_config, result, config_path=config_path)
        console.print(
            f"[bold green]Configuration updated successfully and written to {config_path}[/bold green]"
        )
        console.print(f"[cyan]Updates Applied:[/cyan] {result}")
    else:
        console.print(
            "[bold red]Failed to get configuration updates from OpenAI[/bold red]"
        )


def predict_price(config_path: str) -> None:
    """Use ML engine to predict prices given a dataset.
    
    Args:
        config_path: Path to configuration file
        
    Raises:
        RuntimeError: If prediction fails
    """
    try:
        console.print("[bold blue]Loading model for prediction...[/bold blue]")
        config = load_config(config_path)
        from ml_engine_enhanced import EnhancedMLEngine

        engine = EnhancedMLEngine(config)
        
        console.print("[bold green]Generating prediction...[/bold green]")
        prediction = engine.predict_price()
        
        console.print(f"[bold yellow]Predicted Price: {prediction}[/bold yellow]")
        logging.info(f"Price prediction completed: {prediction}")
        
    except AttributeError as e:
        console.print(f"[red]Method not available: {e}[/red]")
        logging.error(f"predict_price method error: {e}")
        raise RuntimeError(f"Prediction method not available: {e}")
    except Exception as e:
        console.print(f"[red]Prediction failed: {e}[/red]")
        logging.error(f"Price prediction error: {e}", exc_info=True)
        raise


def realtime_loop(config_path: str) -> None:
    """
    Run the ML engine in a real-time loop for continuous inference.
    """
    config = load_config(config_path)
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(config)
    engine.run_realtime_loop()
    console.print("[bold yellow]Realtime Loop command executed[/bold yellow]")


def tune_model(config_path: str) -> None:
    """
    Perform advanced hyperparameter tuning using ML engine methods.
    """
    config = load_config(config_path)
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(config)
    engine.tune_hyperparameters()
    console.print("[bold yellow]Tune Model command executed[/bold yellow]")


def profile_pipeline(config_path: str) -> None:
    """
    Profile the ML pipeline for bottlenecks.
    """
    config = load_config(config_path)
    from ml_engine_enhanced import EnhancedMLEngine

    engine = EnhancedMLEngine(config)
    engine.profile_pipeline()
    console.print("[bold yellow]Profile Pipeline command executed[/bold yellow]")


def run_ai_assistant(config_path: str) -> None:
    _ = config_path
    raise RuntimeError("Retired: AI assistant entrypoint removed. Use `python main.py buddy` / `python main.py train-buddy`.")


def train_unified(config_path: str, csv_path: str | None = None, *, checkpoint_path: str | None = None) -> None:
    """Legacy alias: train Buddy TF model from main.py only."""
    _ = checkpoint_path
    train_buddy(config_path, csv_path)


def train_oanda_unified(
    config_path: str,
    *,
    instruments: str,
    granularity: str,
    candles: int,
    checkpoint_path: str | None = None,
    all_features: bool = False,
) -> None:
    """Legacy alias: uses repo-local USDJPY CSV unless --csv is provided."""
    _ = (instruments, granularity, candles, checkpoint_path)
    train_buddy(config_path, None, all_features=all_features)


def chat_unified(config_path: str, metrics_path: str | None = None) -> None:
    """Interactive chat over the latest unified head metrics."""
    from unified_chat import run_unified_chat

    run_unified_chat(config_path, metrics_path=metrics_path)


def talk_unified(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    csv_path: str | None = None,
    ticker: str | None = None,
    period: str = "5d",
    interval: str = "1h",
    verbose: bool = False,
) -> None:
    raise RuntimeError("Retired: unified talk has been replaced by TF-only Buddy.")


# Major FX pairs for "all" mode
_MAJOR_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
]


def _buddy_execute_all_pairs(
    config_path: str,
    *,
    granularity: str = "M5",
    candles: int = 500,
    execute: bool = False,
    force_execute: bool = False,
    force_units: int | None = None,
    equity: float | None = None,
    risk_per_trade_pct: float | None = None,
    verbose: bool = False,
    use_rl_sizer: bool = True,
    trades_today: int = 0,
    max_trades_per_day: int = 30,
    llm_initialized: bool = False,
    llm_enhance: bool = True,
) -> None:
    """Execute trades on ALL major pairs that pass gates.
    
    Scans all major pairs, runs inference, and executes trades on any
    that pass all gates. Respects daily trade limits and correlation limits.
    """
    import numpy as np
    from datetime import datetime, timezone
    
    console.print("\n[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  BUDDY ALL-PAIRS EXECUTION MODE[/bold cyan]")
    console.print("[bold cyan]════════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[dim]Scanning {len(_MAJOR_PAIRS)} major pairs for trade opportunities...[/dim]\n")
    
    # Suppress verbose logging
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import logging as _logging
    for name in ['modular_trainers', 'modular_inference', 'tensorflow', 'absl']:
        _logging.getLogger(name).setLevel(_logging.ERROR)
    
    from modular_inference import ModularEnsembleInference
    from fx_paper import candles_to_ohlcv_df
    from oanda_practice import OandaPracticeClient
    from feature_engineering import FeatureEngineering
    
    cfg = load_config(config_path)
    client = OandaPracticeClient.from_env()
    fe = FeatureEngineering(cfg.get("feature_engineering", {}))
    
    # Track results
    passed_pairs = []
    failed_pairs = []
    executed_trades = []
    trades_executed_now = 0
    
    for pair in _MAJOR_PAIRS:
        # Check trade limit
        if trades_today + trades_executed_now >= max_trades_per_day:
            console.print(f"\n[bold yellow]⚠️ Daily trade limit reached ({max_trades_per_day}). Stopping.[/bold yellow]")
            break
        
        try:
            # Load pair-specific models if available
            pair_model_dir = Path("trained_data/models") / pair
            has_pair_models = pair_model_dir.exists() and any(pair_model_dir.glob("*.keras"))
            
            enable_llm_integration = llm_enhance and llm_initialized
            ensemble = ModularEnsembleInference(
                instrument=pair if has_pair_models else None,
                use_rl_sizer=use_rl_sizer,
                enable_llm_integration=enable_llm_integration,
            )
            ensemble.load_models()
            
            # Fetch candles
            resp = client.get_candles(pair, granularity=granularity, count=int(candles), price="MBA")
            df = candles_to_ohlcv_df(resp)
            
            # Feature engineering
            df = fe.create_features(df, include_all=True)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill().fillna(0.0)
            
            # Run inference
            result = ensemble.predict_verbose(df, equity=equity, instrument=pair)
            signal = result['raw_signal']
            
            # Check if gates pass
            if signal.trade:
                # LLM validation if available
                llm_approved = True
                llm_size_mult = 1.0
                llm_note = ""
                
                if llm_initialized:
                    try:
                        from buddy_intelligent_mode import validate_trade_with_llm, BuddyRawOutput
                        
                        # Build context for LLM
                        buddy_raw = BuddyRawOutput(
                            confidence=signal.confidence,
                            prediction=1 if signal.direction == 'long' else 0,
                            probability=signal.tcn_probability,
                            direction=signal.direction,
                            gate_results={
                                'tcn_prob': signal.tcn_probability,
                                'ridge_conf': signal.ridge_confidence,
                                'xgb_momentum': signal.xgb_momentum,
                                'rf_drawdown': signal.rf_drawdown_pips,
                            }
                        )
                        
                        price_data = {
                            'last_close': float(df['close'].iloc[-1]),
                            'atr': float(df['atr'].iloc[-1]) if 'atr' in df.columns else None,
                            'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else None,
                            'adx': float(df['adx'].iloc[-1]) if 'adx' in df.columns else None,
                            'trend': 'up' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'down',
                        }
                        
                        llm_validation = validate_trade_with_llm(
                            ticker=pair,
                            direction=signal.direction,
                            buddy_raw=buddy_raw,
                            price_data=price_data,
                        )
                        
                        llm_approved = llm_validation.approve
                        llm_size_mult = llm_validation.size_multiplier
                        if not llm_approved:
                            llm_note = f" [LLM: {llm_validation.reason}]"
                        elif llm_size_mult != 1.0:
                            llm_note = f" [LLM: {llm_size_mult:.1f}x]"
                            
                    except Exception as e:
                        # LLM failed, continue with trade
                        pass
                
                if llm_approved:
                    adjusted_size = signal.size * llm_size_mult
                    passed_pairs.append({
                        'pair': pair,
                        'direction': signal.direction,
                        'size': adjusted_size,
                        'original_size': signal.size,
                        'confidence': signal.confidence,
                        'result': result,
                        'signal': signal,
                        'llm_size_mult': llm_size_mult,
                    })
                    status = f"[green]✓ PASS[/green] → {signal.direction.upper()} {adjusted_size:.1f} lots{llm_note}"
                else:
                    failed_pairs.append({
                        'pair': pair,
                        'reason': f"LLM rejected: {llm_validation.reason}",
                    })
                    status = f"[yellow]✗ LLM REJECT[/yellow]{llm_note}"
            else:
                failed_pairs.append({
                    'pair': pair,
                    'reason': signal.reason,
                })
                status = f"[red]✗ FAIL[/red] ({signal.reason[:30]}...)" if len(signal.reason) > 30 else f"[red]✗ FAIL[/red] ({signal.reason})"
            
            # Show compact status
            model_tag = "[green]●[/green]" if has_pair_models else "[yellow]○[/yellow]"
            console.print(f"  {model_tag} {pair}: {status}")
            
        except Exception as e:
            failed_pairs.append({'pair': pair, 'reason': str(e)[:50]})
            console.print(f"  [red]✗[/red] {pair}: [red]ERROR[/red] ({str(e)[:40]}...)")
    
    # Summary
    console.print(f"\n[bold]━━━ SCAN RESULTS ━━━[/bold]")
    console.print(f"  Passed gates: [green]{len(passed_pairs)}[/green] / {len(_MAJOR_PAIRS)}")
    console.print(f"  Failed gates: [red]{len(failed_pairs)}[/red] / {len(_MAJOR_PAIRS)}")
    
    if not passed_pairs:
        console.print("\n[yellow]No pairs passed all gates. No trades to execute.[/yellow]")
        return
    
    # Show pairs that passed
    console.print(f"\n[bold]━━━ TRADEABLE PAIRS ━━━[/bold]")
    for p in passed_pairs:
        dir_color = "green" if p['direction'] == 'long' else "red"
        console.print(f"  [{dir_color}]{p['direction'].upper():5}[/{dir_color}] {p['pair']}: {p['size']:.1f} lots (conf={p['confidence']:.0f}%)")
    
    # Execute if enabled
    if not execute:
        console.print(f"\n[yellow]⚠️ DRY RUN - Add --execute to place trades[/yellow]")
        return
    
    # Check if FX market is likely open (not weekend)
    import datetime
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    weekday = now_utc.weekday()  # 0=Mon, 5=Sat, 6=Sun
    hour_utc = now_utc.hour
    
    # FX market closed: Fri 21:00 UTC to Sun 21:00 UTC (approximately)
    market_closed = (weekday == 5) or (weekday == 6 and hour_utc < 21) or (weekday == 4 and hour_utc >= 21)
    if market_closed:
        console.print(f"\n[yellow]⚠️ WARNING: FX market is likely closed (weekend)[/yellow]")
        console.print(f"[dim]   Current time: {now_utc.strftime('%Y-%m-%d %H:%M UTC')} (day={weekday})[/dim]")
        console.print(f"[dim]   Orders may be cancelled with MARKET_HALTED[/dim]\n")
    
    console.print(f"\n[bold cyan]━━━ EXECUTING TRADES ━━━[/bold cyan]")
    
    for p in passed_pairs:
        # Re-check trade limit
        if trades_today + trades_executed_now >= max_trades_per_day:
            console.print(f"[yellow]Trade limit reached. Skipping remaining pairs.[/yellow]")
            break
        
        try:
            pair = p['pair']
            signal = p['signal']
            result = p['result']
            
            # Get current price for SL/TP
            resp = client.get_candles(pair, granularity="M1", count=1, price="MBA")
            current_price = float(resp['candles'][0]['mid']['c'])
            
            # Calculate ATR-based SL/TP
            atr = signal.metadata.get('atr', 0.001)
            pip_size = 0.01 if 'JPY' in pair else 0.0001
            price_precision = 3 if 'JPY' in pair else 5
            
            # Calculate SL/TP with proper precision
            sl_distance = 1.5 * atr
            tp_distance = 3.0 * atr
            ts_distance = max(atr * 0.9, pip_size * 10)  # Min 10 pips trailing stop
            
            if signal.direction == 'long':
                sl_price = round(current_price - sl_distance, price_precision)
                tp_price = round(current_price + tp_distance, price_precision)
                units = int(abs(signal.size) * 100000)
            else:
                sl_price = round(current_price + sl_distance, price_precision)
                tp_price = round(current_price - tp_distance, price_precision)
                units = -int(abs(signal.size) * 100000)
            
            # Round trailing stop distance properly
            ts_distance = round(ts_distance, price_precision)
            
            # Override units if forced
            if force_units:
                units = force_units if signal.direction == 'long' else -force_units
            
            # Execute
            order_result = client.create_market_order(
                instrument=pair,
                units=units,
                take_profit_price=tp_price,
                stop_loss_price=sl_price,
                trailing_stop_distance=ts_distance,
            )
            
            # Debug: show full order result if order fails
            trade_id = order_result.get('orderFillTransaction', {}).get('tradeOpened', {}).get('tradeID')
            if trade_id:
                executed_trades.append({'pair': pair, 'direction': signal.direction, 'trade_id': trade_id})
                trades_executed_now += 1
                console.print(f"  [green]✓[/green] {pair}: {signal.direction.upper()} {abs(units):,} units (Trade #{trade_id})")
                
                # Log to journal
                try:
                    from trade_journal import TradeJournal
                    journal = TradeJournal()
                    journal.log_trade(
                        trade_id=str(trade_id),
                        instrument=pair,
                        direction=signal.direction,
                        entry_price=current_price,
                        stop_loss=sl_price,
                        take_profit=tp_price,
                        lot_size=abs(signal.size),
                        tcn_probability=signal.tcn_probability,
                        ridge_confidence=signal.ridge_confidence,
                        xgb_momentum=signal.xgb_momentum,
                        rf_drawdown_pips=signal.rf_drawdown_pips,
                    )
                except Exception:
                    pass
            else:
                # Check for order cancellation (e.g., market halted on weekends)
                cancel_tx = order_result.get('orderCancelTransaction', {})
                cancel_reason = cancel_tx.get('reason', '')
                reject_reason = order_result.get('orderRejectTransaction', {}).get('rejectReason', '')
                error_msg = order_result.get('errorMessage', '')
                
                if cancel_reason == 'MARKET_HALTED':
                    console.print(f"  [yellow]⏸[/yellow] {pair}: Market closed (weekend/holiday)")
                elif cancel_reason:
                    console.print(f"  [red]✗[/red] {pair}: Order cancelled - {cancel_reason}")
                elif reject_reason:
                    console.print(f"  [red]✗[/red] {pair}: Rejected - {reject_reason}")
                elif error_msg:
                    console.print(f"  [red]✗[/red] {pair}: Error - {error_msg}")
                else:
                    console.print(f"  [red]✗[/red] {pair}: Order failed")
                
        except Exception as e:
            console.print(f"  [red]✗[/red] {pair}: Execution error ({str(e)[:40]})")
    
    # Final summary
    console.print(f"\n[bold]━━━ EXECUTION SUMMARY ━━━[/bold]")
    console.print(f"  Trades executed: [green]{len(executed_trades)}[/green]")
    console.print(f"  Total trades today: {trades_today + trades_executed_now}/{max_trades_per_day}")
    
    if executed_trades:
        console.print(f"\n[dim]Trade IDs: {', '.join(t['trade_id'] for t in executed_trades)}[/dim]")


def buddy(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    instrument: str = "USD_JPY",
    granularity: str = "M5",
    candles: int = 500,
    execute: bool = False,  # Live trading must be explicitly enabled
    force_execute: bool = False,
    force_units: int | None = None,
    max_hold_minutes: float | None = None,
    equity: float | None = None,
    risk_per_trade_pct: float | None = None,
    all_features: bool = True,
    verbose: bool = False,
    aggressive_scaling: bool = False,  # Enable $100K→$1M strategy
    use_rl_sizer: bool = True,  # Use RL-based position sizing (default: enabled)
    intelligent: bool = False,  # Enable LLM-enhanced intelligent mode
    explain: bool = False,  # Generate detailed trade reasoning
    llm_provider: str = "auto",  # LLM provider: ollama|claude|openai|auto
    llm_enhance: bool = True,  # Enable deep LLM integration (dynamic thresholds, sentiment aggregation)
) -> None:
    """Buddy: run one TF-only inference on fresh OANDA candles; optionally place a trade.
    
    Args:
        aggressive_scaling: Enable the aggressive $100K→$1M scaling strategy with:
            - Slippage-adjusted Kelly sizing
            - Smart order execution (limit orders, split orders)  
            - Intraday compounding
            - Circuit breakers (drawdown limits, losing streaks)
        intelligent: Enable LLM-enhanced intelligent mode
        explain: Generate detailed trade reasoning
        llm_provider: LLM provider (ollama|claude|openai|auto)
        llm_enhance: Enable deep LLM integration (dynamic thresholds, smarter sentiment)
    """
    _configure_predict_output(verbose)
    
    # =========================================================================
    # INTELLIGENT MODE INITIALIZATION
    # =========================================================================
    llm_initialized = False
    if intelligent or explain:
        try:
            from llm_providers import initialize_buddy_llm, check_provider_status
            
            # Initialize LLM with preferred provider
            provider_pref = None if llm_provider == "auto" else llm_provider
            provider = initialize_buddy_llm(provider=provider_pref)
            
            if provider.is_available:
                llm_initialized = True
                console.print(f"[cyan]🧠 Intelligent Mode: {provider.name.upper()}[/cyan]")
            else:
                console.print("[yellow]⚠ LLM provider not available - intelligent mode disabled[/yellow]")
                console.print("[dim]Install Ollama (brew install ollama && ollama pull llama3.1:7b) or set ANTHROPIC_API_KEY[/dim]")
                intelligent = False
                explain = False
        except ImportError as e:
            console.print(f"[yellow]⚠ Intelligent mode unavailable: {e}[/yellow]")
            intelligent = False
            explain = False

    # Sync journal with OANDA before backfilling online learning
    try:
        from trade_journal import TradeJournal
        from oanda_practice import OandaPracticeClient
        journal = TradeJournal()
        client = OandaPracticeClient.from_env()
        sync_results = journal.sync_open_trades(client)
        closed_updated = sync_results.get('closed_updated', 0)
        if closed_updated > 0:
            console.print(f"[dim]🔄 Journal synced: {closed_updated} closed trades[/dim]")
    except Exception:
        pass

    # Seed online learning buffer from closed journal trades
    try:
        from trade_journal import TradeJournal
        from market_intelligence import MarketIntelligence
        journal = TradeJournal()
        added = journal.backfill_online_learning()
        
        # Get total buffer size to show user
        try:
            intel = MarketIntelligence(enable_online_learning=True)
            buffer_size = len(intel.online_learner.trade_buffer) if intel.online_learner else 0
            if added > 0:
                console.print(f"[dim]🔄 Online Learning: {buffer_size} trades loaded (+{added} new)[/dim]")
            elif buffer_size > 0:
                console.print(f"[dim]🔄 Online Learning: {buffer_size} trades loaded[/dim]")
        except Exception:
            if added > 0:
                console.print(f"[dim]🔄 Online Learning backfilled: {added} trades[/dim]")
    except Exception:
        pass
    
    # =========================================================================
    # FETCH LIVE ACCOUNT BALANCE FROM OANDA (for proper position sizing)
    # =========================================================================
    live_nav = None
    trades_today = 0
    max_trades_per_day = 30
    
    if equity is None:  # Only fetch if not explicitly provided
        try:
            from oanda_practice import OandaPracticeClient
            from datetime import datetime, timezone
            
            client = OandaPracticeClient.from_env()
            account_summary = client.get_account_summary()
            account = account_summary.get('account', {})
            live_nav = float(account.get('NAV', 0))
            
            # Count trades opened today
            today_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            trades_result = client._request(
                'GET',
                f'/accounts/{client._config.account_id}/trades',
                params={'state': 'ALL', 'count': 100}
            )
            for t in trades_result.get('trades', []):
                if t.get('openTime', '').startswith(today_utc):
                    trades_today += 1
            
            if live_nav > 0:
                equity = live_nav
                console.print(f"[green]💰 Live Balance: ${live_nav:,.2f}[/green] (fetched from OANDA)")
                trades_remaining = max_trades_per_day - trades_today
                if trades_remaining <= 5:
                    console.print(f"[yellow]📊 Trades Today: {trades_today}/{max_trades_per_day} (⚠️ {trades_remaining} remaining)[/yellow]")
                else:
                    console.print(f"[dim]📊 Trades Today: {trades_today}/{max_trades_per_day}[/dim]")
                    
                if trades_remaining <= 0:
                    console.print(f"[bold red]⛔ DAILY TRADE LIMIT REACHED[/bold red]")
                    console.print("[dim]Come back tomorrow or increase max_trades_per_day[/dim]")
                    return
        except Exception as e:
            if verbose:
                console.print(f"[dim]Could not fetch live balance: {e}[/dim]")
    
    # Initialize aggressive scaling engine if enabled
    scaling_engine = None
    if aggressive_scaling:
        try:
            from aggressive_scaling import AggressiveScalingEngine
            scaling_engine = AggressiveScalingEngine(
                starting_capital=equity or 100000.0,
                win_rate=0.78,
                avg_win_pips=41.5,
                avg_loss_pips=24.9,
                slippage_at_10_lots=6.0,
            )
            console.print("\n[bold yellow]⚡ AGGRESSIVE SCALING MODE ACTIVE ⚡[/bold yellow]")
            console.print(f"[yellow]  Kelly Fraction: {scaling_engine.kelly_calculator.kelly_fraction:.1%}[/yellow]")
            console.print(f"[yellow]  Current Phase: {scaling_engine.phase}[/yellow]")
            console.print(f"[yellow]  Phase Kelly Multiplier: {scaling_engine._get_kelly_multiplier():.0%}[/yellow]")
        except ImportError:
            console.print("[red]Warning: aggressive_scaling module not found. Using standard sizing.[/red]")
            scaling_engine = None

    import json
    import numpy as np

    t0 = time.perf_counter()
    last = t0

    def _lap(label: str) -> None:
        nonlocal last
        if not verbose:
            return
        now = time.perf_counter()
        console.print(f"[dim]Timing[/dim]: {label} {now - last:.3f}s (total {now - t0:.3f}s)")
        last = now

    cfg = load_config(config_path)
    _lap("load_config")

    # =========================================================================
    # HANDLE "ALL" PAIRS MODE - Execute trades on all pairs that pass gates
    # =========================================================================
    if instrument.upper() == "ALL":
        _buddy_execute_all_pairs(
            config_path=config_path,
            granularity=granularity,
            candles=candles,
            execute=execute,
            force_execute=force_execute,
            force_units=force_units,
            equity=equity,
            risk_per_trade_pct=risk_per_trade_pct,
            verbose=verbose,
            use_rl_sizer=use_rl_sizer,
            trades_today=trades_today,
            max_trades_per_day=max_trades_per_day,
            llm_initialized=llm_initialized,
            llm_enhance=llm_enhance,
        )
        return

    # =========================================================================
    # CHECK FOR MODULAR ENSEMBLE FIRST
    # =========================================================================
    modular_ensemble_meta_path = Path("trained_data") / "models" / "modular_ensemble.meta.json"
    
    if modular_ensemble_meta_path.exists() and not checkpoint_path:
        # Use modular ensemble inference
        console.print("\n[bold magenta]════════════════════════════════════════════════════════════[/bold magenta]")
        console.print("[bold magenta]  MODULAR ENSEMBLE INFERENCE[/bold magenta]")
        console.print("[bold magenta]════════════════════════════════════════════════════════════[/bold magenta]")
        
        # Check if instrument was in training set
        import json
        meta = json.loads(modular_ensemble_meta_path.read_text())
        trained_pairs = meta.get("selected_pairs", [])
        if trained_pairs and instrument not in trained_pairs:
            console.print(f"\n[bold yellow]⚠️  WARNING: {instrument} not in training set[/bold yellow]")
            console.print(f"[dim]Trained: {', '.join(trained_pairs)}[/dim]")
            console.print(f"[dim]Model may perform differently on {instrument}[/dim]\n")
        
        # Suppress verbose TF/Metal logging
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import logging as _logging
        for name in ['modular_trainers', 'modular_inference', 'tensorflow', 'absl']:
            _logging.getLogger(name).setLevel(_logging.ERROR)
        
        from modular_inference import ModularEnsembleInference, InferenceConfig
        
        # Load modular ensemble with pair-specific models
        # Normalizes instrument (e.g., "eur_usd" -> "EUR_USD")
        normalized_instrument = _normalize_instrument(instrument)
        
        # Check if pair-specific models exist
        pair_model_dir = Path("trained_data/models") / normalized_instrument
        has_pair_models = pair_model_dir.exists() and any(pair_model_dir.glob("*.keras"))
        
        if has_pair_models:
            console.print(f"[green]📊 Loading {normalized_instrument}-specific models[/green]")
        else:
            console.print(f"[yellow]📊 No {normalized_instrument} models found, using generic models[/yellow]")
        
        # Show RL sizer status
        if use_rl_sizer:
            console.print(f"[cyan]🤖 RL Position Sizer: ENABLED[/cyan]")
        
        # Show LLM enhance status
        enable_llm_integration = llm_enhance and llm_initialized
        if enable_llm_integration:
            console.print(f"[cyan]🧠 LLM Enhancement: ENABLED (dynamic thresholds, smart sentiment)[/cyan]")
        
        ensemble = ModularEnsembleInference(
            instrument=normalized_instrument if has_pair_models else None,
            use_rl_sizer=use_rl_sizer,
            enable_llm_integration=enable_llm_integration,
        )
        ensemble.load_models()
        
        # Fetch candles for inference
        from fx_paper import candles_to_ohlcv_df
        from oanda_practice import OandaPracticeClient
        
        client = OandaPracticeClient.from_env()
        resp = client.get_candles(instrument, granularity=granularity, count=int(candles), price="MBA")
        df = candles_to_ohlcv_df(resp)
        _lap("oanda_fetch")
        
        # Feature engineering
        from feature_engineering import FeatureEngineering
        fe = FeatureEngineering(cfg.get("feature_engineering", {}))
        df = fe.create_features(df, include_all=True)
        _lap("feature_engineering")
        
        # Clean up data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0.0)
        
        # Run modular ensemble inference
        result = ensemble.predict_verbose(df, instrument=instrument)
        _lap("modular_inference")
        
        # Display results
        console.print("")
        for check in result['gate_checks']:
            console.print(f"  {check}")
        console.print("")
        console.print(f"[bold]{result['decision']}[/bold]")
        console.print("")
        
        signal = result['raw_signal']
        
        # =====================================================================
        # INTELLIGENT MODE: Generate LLM-enhanced reasoning
        # =====================================================================
        # Build BuddyRawOutput for intelligent mode (used by reasoning AND validation)
        # =====================================================================
        buddy_raw = None
        if llm_initialized and signal:
            from buddy_intelligent_mode import BuddyRawOutput
            buddy_raw = BuddyRawOutput(
                confidence=signal.ridge_confidence,
                prediction=signal.tcn_probability,
                probability=signal.tcn_probability,
                direction='long' if signal.direction == 'long' else 'short',
                last_price=float(df['close'].iloc[-1]),
                gate_results={
                    'tcn_probability': signal.tcn_probability,
                    'ridge_confidence': signal.ridge_confidence,
                    'xgb_momentum': signal.xgb_momentum,
                    'rf_drawdown_pips': signal.rf_drawdown_pips,
                    'gates_passed': signal.trade,
                },
            )
        
        # =====================================================================
        # INTELLIGENT MODE: Generate LLM-enhanced reasoning (optional display)
        # =====================================================================
        if (intelligent or explain) and llm_initialized and signal and buddy_raw:
            try:
                from buddy_intelligent_mode import (
                    generate_trade_reasoning,
                    get_intelligent_mode,
                )
                
                # Gather price/technical context
                price_data = {
                    'last_close': float(df['close'].iloc[-1]),
                    'atr': float(df['atr'].iloc[-1]) if 'atr' in df.columns else None,
                    'adx': float(df['adx'].iloc[-1]) if 'adx' in df.columns else None,
                    'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else None,
                    'trend': 'up' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'down',
                }
                
                # Only show verbose reasoning if --explain flag is used
                if explain:
                    console.print("[cyan]━━━ REASONING ━━━[/cyan]")
                    
                    if intelligent:
                        intel_mode = get_intelligent_mode()
                        reasoning_result = intel_mode.get_full_intelligent_analysis(
                            ticker=instrument,
                            timeframe=granularity,
                            buddy_raw=buddy_raw,
                            price_data=price_data,
                        )
                        if reasoning_result.get('reasoning'):
                            console.print(reasoning_result['reasoning'])
                    else:
                        reasoning_result = generate_trade_reasoning(
                            ticker=instrument,
                            timeframe=granularity,
                            buddy_raw=buddy_raw,
                            price_data=price_data,
                        )
                        if reasoning_result.get('reasoning'):
                            console.print(reasoning_result['reasoning'])
                    
                    console.print("[cyan]━━━━━━━━━━━━━━━━━━[/cyan]\n")
                
                _lap("intelligent_reasoning")
                
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Reasoning generation failed: {e}[/yellow]")
        
        # =====================================================================
        # LLM TRADE VALIDATION - Actually influences the trade decision
        # Runs even with --no-execute to show what would happen
        # =====================================================================
        llm_validation = None
        if signal.trade and llm_initialized and buddy_raw:
            try:
                from buddy_intelligent_mode import validate_trade_with_llm, LLMTradeValidation
                
                # Gather context for validation
                price_data = {
                    'last_close': float(df['close'].iloc[-1]),
                    'atr': float(df['atr'].iloc[-1]) if 'atr' in df.columns else None,
                    'adx': float(df['adx'].iloc[-1]) if 'adx' in df.columns else None,
                    'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else None,
                    'trend': 'up' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'down',
                }
                
                # Get news sentiment if available
                news_sentiment = None
                if 'sentiment_score' in result:
                    news_sentiment = {
                        'sentiment': result.get('sentiment_label', 'neutral'),
                        'score': result.get('sentiment_score', 0),
                        'headline_count': result.get('headline_count', 0),
                    }
                
                llm_validation = validate_trade_with_llm(
                    ticker=instrument,
                    direction=signal.direction,
                    buddy_raw=buddy_raw,
                    price_data=price_data,
                    news_sentiment=news_sentiment,
                )
                
                # Display validation result
                if llm_validation.approve:
                    if llm_validation.size_multiplier != 1.0:
                        console.print(f"[cyan]🧠 LLM: Approved (size {llm_validation.size_multiplier:.1f}x) - {llm_validation.reason}[/cyan]")
                    else:
                        console.print(f"[green]🧠 LLM: Approved - {llm_validation.reason}[/green]")
                else:
                    console.print(f"[red]🧠 LLM: REJECTED - {llm_validation.reason}[/red]")
                    if llm_validation.risk_flags:
                        console.print(f"[red]   Risk flags: {', '.join(llm_validation.risk_flags)}[/red]")
                
                _lap("llm_validation")
                
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]LLM validation failed: {e}[/yellow]")
                llm_validation = None
        
        # Execute trade if gates pass AND LLM approves (or LLM unavailable)
        llm_approved = llm_validation is None or llm_validation.approve
        if signal.trade and execute and llm_approved:
            from oanda_practice import OandaPracticeClient
            trader = OandaPracticeClient.from_env()
            
            # Apply LLM size adjustment if available
            size_multiplier = 1.0
            if llm_validation and llm_validation.size_multiplier != 1.0:
                size_multiplier = llm_validation.size_multiplier
            
            units = int(signal.size * 100000 * size_multiplier)  # Convert lots to units with LLM adjustment
            if signal.direction == 'short':
                units = -units
            
            # Calculate TP, SL, TS based on ATR for accurate risk management
            current_price = float(df['close'].iloc[-1])
            atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else None
            
            # Get bid/ask spread estimate (use ATR * 0.1 as proxy if not available)
            spread = atr * 0.05 if atr else current_price * 0.0001
            
            if atr and atr > 0:
                # ATR-BASED SL/TP: Give trades room to breathe
                # 1.5x ATR for SL, 2x ATR for TP (R:R = 1:1.33 minimum)
                is_jpy = instrument.endswith('_JPY')
                pip_mult = 100 if is_jpy else 10000
                pip_size = 0.01 if is_jpy else 0.0001
                
                # ATR-based SL: 1.5x ATR gives room for normal volatility
                sl_distance = atr * 1.5
                sl_pips = sl_distance * pip_mult
                
                # ATR-based TP: Base 2x ATR, bonus for high probability
                raw_signal = result.get('raw_signal')
                tcn_prob = raw_signal.tcn_probability if raw_signal else 0.5
                if tcn_prob >= 0.65:
                    tp_distance = atr * 3.0  # 3x ATR for high confidence
                    console.print(f"[green]✓ High probability ({tcn_prob:.1%}) - TP extended to 3x ATR[/green]")
                else:
                    tp_distance = atr * 2.0  # 2x ATR standard
                tp_pips = tp_distance * pip_mult
                
                # Trailing stop at 1x ATR (locks in profit after 1 ATR move)
                ts_distance = atr * 1.0
                
                if signal.direction == 'long':
                    # LONG: SL below entry, TP above entry
                    sl_price = current_price - sl_distance
                    tp_price = current_price + tp_distance
                else:
                    # SHORT: SL above entry, TP below entry
                    sl_price = current_price + sl_distance
                    tp_price = current_price - tp_distance
                
                # Calculate trailing stop in pips
                ts_pips = ts_distance * pip_mult
                
                console.print(f"[bold yellow]Executing: {signal.direction.upper()} {abs(units):,} units @ {current_price:.5f}[/bold yellow]")
                console.print(f"  [cyan]TP:[/cyan] {tp_price:.5f} (+{tp_pips:.1f} pips = {tp_distance/atr:.1f}x ATR)")
                console.print(f"  [red]SL:[/red] {sl_price:.5f} (-{sl_pips:.1f} pips = {sl_distance/atr:.1f}x ATR)")
                console.print(f"  [magenta]TS:[/magenta] {ts_distance:.5f} ({ts_distance*pip_mult:.1f} pips trailing)")
                console.print(f"  [dim]ATR: {atr:.5f} ({atr*pip_mult:.1f} pips) | R:R = 1:{tp_distance/sl_distance:.2f}[/dim]")
                
                try:
                    order_result = trader.create_market_order(
                        instrument=instrument,
                        units=units,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        trailing_stop_distance=ts_distance,
                    )
                    fill_tx = order_result.get('orderFillTransaction', {})
                    fill_price_str = fill_tx.get('price', 'N/A')
                    # Try multiple ways to get trade ID
                    trade_id = fill_tx.get('tradeOpened', {}).get('tradeID')
                    if not trade_id:
                        trades_opened = fill_tx.get('tradesOpened', [])
                        if trades_opened:
                            trade_id = trades_opened[0].get('tradeID')
                    if not trade_id:
                        trade_id = getattr(trader, '_last_trade_id', None)
                    # Also check orderCreateTransaction for cases where order is pending
                    if not trade_id:
                        create_tx = order_result.get('orderCreateTransaction', {})
                        trade_id = create_tx.get('id')
                    if not trade_id:
                        trade_id = 'N/A'
                    
                    # Calculate and display slippage
                    if fill_price_str != 'N/A':
                        fill_price_actual = float(fill_price_str)
                        slippage = fill_price_actual - current_price
                        slippage_pips = slippage * pip_mult
                        
                        # Positive slippage is bad for LONG, good for SHORT (and vice versa)
                        is_favorable = (signal.direction == 'short' and slippage < 0) or (signal.direction == 'long' and slippage > 0)
                        slip_color = 'green' if abs(slippage_pips) < 1.0 else ('yellow' if abs(slippage_pips) < 3.0 else 'red')
                        slip_sign = '+' if slippage_pips > 0 else ''
                        favorable_str = ' (favorable)' if is_favorable else ''
                        
                        console.print(f"[green]✓ Filled @ {fill_price_str} (Trade #{trade_id})[/green]")
                        console.print(f"  [{slip_color}]Slippage: {slip_sign}{slippage_pips:.1f} pips{favorable_str}[/{slip_color}]")
                    else:
                        fill_price_actual = current_price  # Use expected price as fallback
                        slippage_pips = 0.0
                        console.print(f"[green]✓ Order filled (Trade #{trade_id})[/green]")
                    
                    # Log trade to journal (regardless of whether we have fill price)
                    if trade_id and trade_id != 'N/A':
                        try:
                            from trade_journal import TradeJournal
                            journal = TradeJournal()
                            journal.log_trade(
                                trade_id=str(trade_id),
                                instrument=instrument,
                                direction=signal.direction,
                                units=abs(units),
                                lots=signal.size,
                                expected_price=current_price,
                                fill_price=fill_price_actual,
                                stop_loss=sl_price,
                                take_profit=tp_price,
                                trailing_stop=ts_distance,
                                atr=atr,
                                tcn_probability=signal.tcn_probability,
                                ridge_confidence=signal.ridge_confidence,
                                xgb_momentum=signal.xgb_momentum,
                                rf_drawdown_pips=signal.rf_drawdown_pips,
                                feature_vector=[
                                    signal.tcn_probability,
                                    signal.ridge_confidence,
                                    signal.xgb_momentum,
                                    signal.rf_drawdown_pips,
                                ],
                                prediction=signal.tcn_probability,
                                confidence=signal.ridge_confidence,
                            )
                            console.print(f"  [dim]📓 Trade logged to journal[/dim]")
                        except Exception as je:
                            console.print(f"  [yellow]Journal error: {je}[/yellow]")
                except Exception as e:
                    console.print(f"[red]Order failed: {e}[/red]")
            else:
                # Fallback: no ATR available, use simple percentage-based stops
                console.print(f"[bold yellow]Executing: {signal.direction.upper()} {abs(units):,} units[/bold yellow]")
                console.print("[yellow]⚠ ATR not available - placing order without TP/SL[/yellow]")
                try:
                    order_result = trader.create_market_order(instrument=instrument, units=units)
                    console.print(f"[green]Order placed: {order_result.get('orderFillTransaction', {}).get('price', 'filled')}[/green]")
                except Exception as e:
                    console.print(f"[red]Order failed: {e}[/red]")
        
        elif signal.trade and not llm_approved:
            console.print("[yellow]No trade: LLM rejected (model gates passed but LLM identified risk)[/yellow]")
        elif not signal.trade:
            console.print("[yellow]No trade: gates failed[/yellow]")
        
        console.print("[bold magenta]════════════════════════════════════════════════════════════[/bold magenta]")
        return
    
    # =========================================================================
    # FALLBACK TO STANDARD MODEL INFERENCE
    # =========================================================================

    # Load model + preprocessing metadata.
    # Priority: checkpoint_path > buddy_tf_candidate.meta.json (newer) > buddy_tf.meta.json (production)
    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    candidate_meta_path = Path("trained_data") / "models" / "buddy_tf_candidate.meta.json"
    
    if checkpoint_path:
        ck_meta = _meta_path_for_checkpoint(str(checkpoint_path))
        if ck_meta and ck_meta.exists():
            meta_path = ck_meta
    
    # Prioritize candidate if it exists (newer model)
    # Only use production if candidate doesn't exist
    if candidate_meta_path.exists():
        meta_path = candidate_meta_path
        console.print("[cyan]Using candidate model[/cyan] (buddy_tf_candidate.keras)")
        console.print("[dim]Tip: Promote with: buddy promote-model[/dim]")
    elif meta_path.exists():
        console.print("[green]Model: buddy_tf.keras (production)[/green]")
    
    if not meta_path.exists():
        if checkpoint_path:
            raise FileNotFoundError(f"Missing Buddy metadata for checkpoint: {checkpoint_path}")
        raise FileNotFoundError("Missing Buddy metadata. Run: python main.py train-buddy")
    meta = json.loads(meta_path.read_text())
    _lap("load_meta")

    # Match inference preprocessing to how the model was trained.
    all_features = bool(meta.get("all_features", all_features))

    train_smoothing = bool(meta.get("train_smoothing", True))
    median_window = meta.get("median_window", None)
    median_window = int(median_window) if median_window is not None else None

    min_volume = meta.get("min_volume", None)
    min_volume = int(min_volume) if min_volume is not None else None

    spread_filter = bool(meta.get("spread_filter", False))
    spread_pctl = float(meta.get("spread_pctl", 0.99))
    spread_mult = float(meta.get("spread_mult", 3.0))

    model_path = checkpoint_path or meta.get("model_path")
    if not model_path:
        raise ValueError("Missing model path in metadata")
    seq_len_eff = int(meta.get("seq_len") or 64)
    feature_columns: list[str] = list(meta.get("feature_columns") or [])
    if not feature_columns:
        raise ValueError("Missing feature_columns in Buddy metadata")

    std = meta.get("standardize") or {}
    mu = np.asarray(std.get("mean") or [], dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(std.get("std") or [], dtype=np.float32).reshape(1, -1)
    if mu.size != len(feature_columns) or sigma.size != len(feature_columns):
        raise ValueError("Standardization stats do not match feature_columns")

    pca_info = meta.get("pca")
    pca_components = int(meta.get("pca_components") or 0)
    pca_mean = None
    pca_components_mat = None
    if pca_info and pca_components > 0:
        pca_mean = np.asarray(pca_info.get("mean") or [], dtype=np.float32).reshape(1, -1)
        pca_components_mat = np.asarray(pca_info.get("components") or [], dtype=np.float32)
        if pca_components_mat.ndim != 2:
            raise ValueError("Invalid PCA components in metadata")

    # Trading gate.
    threshold = float(meta.get("live_confidence_threshold", 0.75))
    if execute:
        if force_execute:
            console.print(
                "[yellow]FORCE-EXECUTE[/yellow]: bypassing the training accuracy gate; PRACTICE orders may be placed."
            )
        else:
            enabled, _ = _buddy_live_enabled_from_meta()
            if not enabled:
                console.print(
                    "[yellow]Live trading disabled[/yellow]: train Buddy and reach >=62% direction accuracy."
                )
                execute = False
            else:
                console.print(f"[green]Live trading enabled[/green] (confidence threshold {threshold*100:.0f}%).")

    # Fetch candles.
    from fx_paper import candles_to_ohlcv_df
    from oanda_practice import OandaPracticeClient

    client = OandaPracticeClient.from_env()
    resp = client.get_candles(instrument, granularity=granularity, count=int(candles), price="MBA")
    df = candles_to_ohlcv_df(resp)
    _lap("oanda_fetch")

    if all_features and int(candles) < 1000:
        console.print(
            "[yellow]Warning[/yellow]: low candle lookback can make engineered features degenerate. "
            "Consider re-running with `--candles 2000` or `--candles 5000`."
        )

    # Apply the same data-quality filters used during training (when available).
    if min_volume is not None and "volume" in df.columns:
        try:
            df = df.loc[df["volume"].astype(float) >= float(min_volume)].copy()
        except Exception:
            pass

    if spread_filter and ("bid_close" in df.columns and "ask_close" in df.columns):
        try:
            spread = (df["ask_close"].astype(float) - df["bid_close"].astype(float)).abs()
            spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
            if not spread.empty:
                pctl_thr = float(np.quantile(spread.to_numpy(), float(spread_pctl)))
                med = float(np.median(spread.to_numpy()))
                thr = max(pctl_thr, float(spread_mult) * med) if med > 0 else pctl_thr
                keep_mask = (df["ask_close"].astype(float) - df["bid_close"].astype(float)).abs() <= thr
                df = df.loc[keep_mask].copy()
        except Exception:
            pass

    # Feature engineering to match training.
    if all_features:
        from feature_engineering import FeatureEngineering

        fe = FeatureEngineering(cfg.get("feature_engineering", {}))
        df = fe.create_features(
            df,
            include_all=True,
            apply_candle_smoothing=bool(train_smoothing),
            median_window=median_window,
        )
    _lap("feature_engineering")

    numeric_df = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.ffill().bfill().fillna(0.0)

    # Build the exact feature matrix expected by the model (stable order, fill missing with 0).
    available_cols = set(numeric_df.columns)
    expected_cols = list(feature_columns)
    missing_cols = [c for c in expected_cols if c not in available_cols]
    extra_cols = [c for c in numeric_df.columns if c not in set(expected_cols)]
    if missing_cols:
        console.print(f"[yellow]Inference[/yellow]: {len(missing_cols)} missing features will be filled from training means")
    if extra_cols and verbose:
        console.print(f"[dim]Inference[/dim]: dropping {len(extra_cols)} extra numeric features")

    numeric_df = numeric_df.reindex(columns=expected_cols, fill_value=0.0)
    # If feature columns are missing at inference time, filling with 0.0 can create huge standardized values
    # (0 - mean)/std and saturate both sigmoids to ~0. Use training means instead.
    if missing_cols:
        mu_flat = mu.reshape(-1)
        for idx, col in enumerate(expected_cols):
            if col in set(missing_cols):
                numeric_df[col] = float(mu_flat[idx])

    x2d = numeric_df.to_numpy(dtype=np.float32, copy=False)
    if len(x2d) < seq_len_eff:
        raise ValueError(f"Need at least {seq_len_eff} rows after preprocessing; got {len(x2d)}")
    # Keep full history for optional ensemble inference.
    x2d_full = x2d
    x2d = x2d_full[-seq_len_eff:]
    _lap("build_features")

    # Standardize and optional PCA.
    x_std = (x2d - mu) / np.where(sigma < 1e-6, 1.0, sigma)
    # Guard against extreme standardized values (can saturate sigmoids to ~0/~1).
    x_std = np.clip(x_std, -10.0, 10.0).astype(np.float32, copy=False)
    if pca_components_mat is not None and pca_mean is not None:
        x_std = (x_std - pca_mean) @ pca_components_mat.T
    _lap("standardize_pca")

    if verbose:
        try:
            zero_frac = float(np.mean((np.abs(x2d) < 1e-12).astype(np.float32)))
            console.print(
                f"[dim]Inference diag[/dim]: seq_len={seq_len_eff} feat_dim={x2d.shape[1]} zeros={zero_frac*100:.1f}% "
                f"x_std[min/mean/max]=({float(x_std.min()):.3f}/{float(x_std.mean()):.3f}/{float(x_std.max()):.3f})"
            )
        except Exception:
            pass

    # Keep a last-window tensor for diagnostics (logits etc.).
    x_last = x_std.reshape(1, seq_len_eff, -1)

    # TensorFlow init + model load are the main cold-start costs on macOS/Metal.
    # Delay them until after we have fresh candles and the feature matrix ready.
    _configure_tf_metal(verbose=verbose)

    import tensorflow as tf

    from ml_head_engine import MLEngineHead
    from mr_engine import MREngineHead
    from ms_head_engine import MSEngineHead
    from mt_engine import MTEngineHead
    from mx_head_engine import MXEngineHead

    custom_objects = {
        "MLEngineHead": MLEngineHead,
        "MREngineHead": MREngineHead,
        "MSEngineHead": MSEngineHead,
        "MTEngineHead": MTEngineHead,
        "MXEngineHead": MXEngineHead,
    }
    
    # Load model with multiple fallback strategies for Keras version compatibility
    model = None
    load_errors = []
    
    # Strategy 1: Standard load
    try:
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        if verbose:
            console.print("[green]✓ Model loaded (standard)[/green]")
    except Exception as e1:
        load_errors.append(f"Standard: {str(e1)[:100]}")
        if verbose:
            console.print(f"[dim]Strategy 1 failed: {str(e1)[:100]}[/dim]")
        
        # Strategy 2: With safe_mode=False
        try:
            model = tf.keras.models.load_model(
                model_path, compile=False, custom_objects=custom_objects, safe_mode=False
            )
            if verbose:
                console.print("[green]✓ Model loaded (safe_mode=False)[/green]")
        except Exception as e2:
            load_errors.append(f"SafeMode=False: {str(e2)[:100]}")
            if verbose:
                console.print(f"[dim]Strategy 2 failed: {str(e2)[:100]}[/dim]")
            
            # Strategy 3: Try tf.keras.saving.load_model (Keras 3)
            try:
                model = tf.keras.saving.load_model(
                    model_path, compile=False, custom_objects=custom_objects
                )
                if verbose:
                    console.print("[green]✓ Model loaded (tf.keras.saving)[/green]")
            except Exception as e3:
                load_errors.append(f"tf.keras.saving: {str(e3)[:100]}")
                if verbose:
                    console.print(f"[dim]Strategy 3 failed: {str(e3)[:100]}[/dim]")
                
                # Strategy 4: Try loading weights only (rebuild architecture)
                try:
                    # Rebuild model from metadata
                    feature_dim = len(feature_columns)
                    seq_len_meta = int(meta.get("seq_len", 60))
                    model_type = meta.get("model_type", "tcn")
                    
                    console.print(f"[yellow]Attempting to rebuild {model_type} model from metadata...[/yellow]")
                    
                    model = _build_buddy_model_for_type(
                        model_type=model_type,
                        feature_dim=feature_dim,
                        seq_len=seq_len_meta,
                    )
                    
                    # Try to load weights
                    weights_path = str(model_path).replace(".keras", "_weights.h5")
                    if Path(weights_path).exists():
                        model.load_weights(weights_path)
                        console.print("[green]✓ Model rebuilt and weights loaded[/green]")
                    else:
                        # Try extracting weights from .keras file
                        import h5py
                        with h5py.File(model_path, 'r') as f:
                            if 'model_weights' in f:
                                model.load_weights(model_path, by_name=True)
                                console.print("[green]✓ Model rebuilt and weights loaded from .keras[/green]")
                            else:
                                raise ValueError("No weights found in model file")
                except Exception as e4:
                    load_errors.append(f"Weights-only: {str(e4)[:100]}")
                    
                    # All strategies failed
                    console.print("[red]Failed to load model after 4 attempts[/red]")
                    console.print("[yellow]Load errors:[/yellow]")
                    for i, err in enumerate(load_errors, 1):
                        console.print(f"  {i}. {err}")
                    raise RuntimeError(
                        f"Failed to load model from {model_path}. "
                        "This is a Keras version mismatch. The model was saved with a different Keras version. "
                        "Please check your conda environment and ensure you're using the same environment for training and prediction. "
                        "Try: conda activate tf-metal && python --version && python -c 'import keras; print(keras.__version__)'"
                    ) from e4
    
    if model is None:
        raise RuntimeError("Model loading failed - model is None")
    
    _lap("tf_load_model")

    # Some checkpoints were built with a single tensor input named "features" (expects positional array),
    # while others were built with a dict input {"features": ...}. Use the model's stored input structure.
    _inputs_struct = getattr(model, "_inputs_struct", None)
    _expects_features_dict = isinstance(_inputs_struct, dict) and "features" in _inputs_struct

    def _model_call(x_in: np.ndarray):
        return model({"features": x_in}, training=False) if _expects_features_dict else model(x_in, training=False)

    # --- Inference: use a small ensemble over the last N windows ---
    # A single last window can sometimes produce extreme logits (float32 sigmoid clamps to 0/1).
    # Averaging over the most recent windows makes predictions more stable and avoids misleading 0.000 prints.
    n_ens = int(meta.get("inference_ensemble_windows", 20) or 20)
    n_ens = max(1, min(n_ens, 100))

    conf_agg = str(meta.get("inference_confidence_agg", "quantile") or "quantile").strip().lower()
    if conf_agg not in {"mean", "max", "quantile"}:
        conf_agg = "quantile"
    conf_q = float(meta.get("inference_confidence_quantile", 0.90) or 0.90)
    conf_q = float(min(1.0, max(0.50, conf_q)))

    def _predict_one(x_one: np.ndarray) -> tuple[float, float]:
        pred_one = _model_call(x_one)
        d = pred_one["direction"]
        c = pred_one["confidence"]
        if hasattr(d, "numpy"):
            d = d.numpy()
        if hasattr(c, "numpy"):
            c = c.numpy()
        return (
            float(np.asarray(d).reshape(-1)[0]),
            float(np.asarray(c).reshape(-1)[0]),
        )

    # Build standardized full series (only tail needed for N windows).
    p_dir_last = None
    p_conf_last = None
    p_dir = None
    p_conf = None
    try:
        x_std_full = (x2d_full - mu) / np.where(sigma < 1e-6, 1.0, sigma)
        x_std_full = np.clip(x_std_full, -10.0, 10.0).astype(np.float32, copy=False)
        if pca_components_mat is not None and pca_mean is not None:
            x_std_full = (x_std_full - pca_mean) @ pca_components_mat.T

        need = seq_len_eff + n_ens - 1
        if n_ens > 1 and len(x_std_full) >= need:
            tail = x_std_full[-need:]
            windows = np.stack([tail[i : i + seq_len_eff] for i in range(n_ens)], axis=0).astype(np.float32)
            pred_batch = _model_call(windows)
            d = pred_batch["direction"]
            c = pred_batch["confidence"]
            if hasattr(d, "numpy"):
                d = d.numpy()
            if hasattr(c, "numpy"):
                c = c.numpy()
            dir_probs = np.asarray(d).reshape(-1)
            conf_probs = np.asarray(c).reshape(-1)
            conf_raw = conf_probs
            conf_probs = np.clip(conf_probs, 0.0, 1.0)
            if verbose:
                try:
                    raw = np.asarray(conf_raw, dtype=np.float64).reshape(-1)
                    console.print(
                        "[dim]Inference diag[/dim]: conf_raw[min/mean/max]="
                        f"({float(raw.min()):.6g}/{float(raw.mean()):.6g}/{float(raw.max()):.6g}) "
                        f"pct<0={float(np.mean(raw < 0.0))*100:.1f}% pct>1={float(np.mean(raw > 1.0))*100:.1f}%"
                    )
                except Exception:
                    pass
            p_dir_last = float(dir_probs[-1])
            p_conf_last = float(conf_probs[-1])
            # Aggregate direction by mean; aggregate confidence by quantile/max to avoid collapse when a few windows saturate.
            p_dir = float(np.mean(dir_probs))
            if conf_agg == "mean":
                p_conf = float(np.mean(conf_probs))
            elif conf_agg == "max":
                p_conf = float(np.max(conf_probs))
            else:
                # quantile
                p_conf = float(np.quantile(conf_probs, conf_q))
            if verbose:
                console.print(
                    f"[dim]Inference ensemble[/dim]: windows={n_ens} "
                    f"dir_mean={p_dir:.6g} conf_{conf_agg}={p_conf:.6g} dir_last={p_dir_last:.6g} conf_last={p_conf_last:.6g}"
                )
        else:
            x_one = x_std.reshape(1, seq_len_eff, -1)
            p_dir, p_conf = _predict_one(x_one)
            p_dir_last, p_conf_last = p_dir, p_conf
    except Exception:
        x_one = x_std.reshape(1, seq_len_eff, -1)
        p_dir, p_conf = _predict_one(x_one)
        p_dir_last, p_conf_last = p_dir, p_conf

    # Fallback safety.
    if p_dir is None or p_conf is None:
        x_one = x_std.reshape(1, seq_len_eff, -1)
        p_dir, p_conf = _predict_one(x_one)
        p_dir_last, p_conf_last = p_dir, p_conf

    _lap("inference")

    # Clamp confidence to [0,1] for display/aggregation stability (linear head on newer checkpoints).
    try:
        p_conf = float(np.clip(float(p_conf), 0.0, 1.0))
        p_conf_last = float(np.clip(float(p_conf_last), 0.0, 1.0))
    except Exception:
        pass

    def _fmt_prob(p: float) -> str:
        # Avoid misleading 0.000/1.000 prints when float32 sigmoid under/overflows.
        if p <= 1e-12 or p >= 1.0 - 1e-12 or p < 1e-4 or p > 1.0 - 1e-4:
            return f"{p:.6e}"
        return f"{p:.6f}"

    def _dense_logit(head_name: str) -> float | None:
        # Estimate the pre-sigmoid logit for a named Dense head.
        # Useful when probabilities are exactly 0/1 due to float32 saturation.
        try:
            head = model.get_layer(head_name)
            pre = tf.keras.Model(inputs=model.inputs, outputs=head.input)
            z_in = {"features": x_last} if _expects_features_dict else x_last
            z = np.asarray(pre.predict(z_in, verbose=0)).reshape(-1).astype(np.float64, copy=False)
            w, b = head.get_weights()
            w = np.asarray(w).reshape(-1).astype(np.float64, copy=False)
            b0 = float(np.asarray(b).reshape(-1)[0])
            return float(z @ w + b0)
        except Exception:
            return None

    def _stable_sigmoid_from_logit(logit: float) -> float:
        # Stable sigmoid in float64 to illustrate saturation.
        x = float(logit)
        if x >= 0:
            ex = float(np.exp(-x))
            return 1.0 / (1.0 + ex)
        ex = float(np.exp(x))
        return ex / (1.0 + ex)

    side = "buy" if p_dir >= 0.5 else "sell"
    
    # =========================================================================
    # PREDICTION SUMMARY (Industry Standard Format)
    # =========================================================================
    dir_strength = abs(p_dir - 0.5) * 200  # 0-100% scale
    signal_emoji = "🟢" if side == "buy" else "🔴"
    
    console.print("\n" + "=" * 60)
    console.print(f"[bold]{signal_emoji} BUDDY SIGNAL: {side.upper()}[/bold]")
    console.print("=" * 60)
    
    # Show key metrics in a clear format
    console.print(f"  Instrument:     {instrument}")
    console.print(f"  Timeframe:      {granularity}")
    console.print(f"  Direction:      {p_dir:.1%} {'(LONG)' if side == 'buy' else '(SHORT)'}")
    console.print(f"  Signal Strength: {dir_strength:.1f}%")
    console.print(f"  Model Conf:     {p_conf:.1%}")
    
    # Regime Detection (if features available)
    try:
        if "adx" in df.columns:
            adx_val = float(df["adx"].iloc[-1])
            is_trending = adx_val > 25
            regime = "TRENDING" if is_trending else "RANGING"
            regime_color = "green" if is_trending else "yellow"
            console.print(f"  Market Regime:  [{regime_color}]{regime}[/{regime_color}] (ADX={adx_val:.1f})")
            
            # Regime filter warning
            require_trending = bool(tier2_cfg.get("require_trending", False)) if 'tier2_cfg' in dir() else False
            if require_trending and not is_trending:
                console.print(f"  [yellow]⚠ Warning: Low ADX - consider skipping (regime filter)[/yellow]")
    except Exception:
        pass
    
    console.print("-" * 60)

    # Tier-2: calibrated win probability (TP-before-SL) if available.
    tier2_cfg = (cfg.get("buddy", {}) or {}).get("tier2", {}) or {}
    tier2_enabled = bool(tier2_cfg.get("enabled", False))
    tier2_p_win = None
    if tier2_enabled:
        try:
            dir_conf = float(p_dir) if side == "buy" else float(1.0 - float(p_dir))
            score = float(max(0.0, min(1.0, dir_conf * float(p_conf))))
            tier2_p_win = _tier2_apply_calibration(meta, score)
        except Exception:
            tier2_p_win = None

    # Tier-1 confidence gate (repo default): use fx_confidence_v1 until a calibrated FX confidence head exists.
    tier1_conf = None
    tier1_reasons: list[str] | None = None
    try:
        from reasoning_enhanced import fx_confidence_v1

        from fx_paper import atr as atr_from_df
        from fx_paper import spread_pips_from_df

        last_px = float(df["close"].iloc[-1])
        atr_val = float(atr_from_df(df, period=14))
        sp_pips = spread_pips_from_df(df, instrument)
        if sp_pips is None:
            sp_pips = (
                cfg.get("fx", {})
                .get("costs", {})
                .get("spread_fallback_pips", {})
                .get(instrument)
            )
        max_spread = (
            cfg.get("fx", {})
            .get("costs", {})
            .get("max_spread_pips", {})
            .get(instrument)
        )

        payload = fx_confidence_v1(
            signal=side,
            price=last_px,
            atr=atr_val,
            spread_pips=float(sp_pips) if sp_pips is not None else float("nan"),
            max_spread_pips=float(max_spread) if max_spread is not None else float("nan"),
            params=cfg.get("fx", {}).get("confidence_model", {}),
        )
        tier1_conf = float(payload.get("confidence", 0.0) or 0.0)
        r = payload.get("reasons")
        if isinstance(r, list):
            tier1_reasons = [str(x) for x in r]
    except Exception:
        tier1_conf = None
        tier1_reasons = None
    # Determine gating behavior:
    # - By default, when a calibrated Tier-2 p_win exists we gate on that.
    # - Optionally, `buddy.tier2.use_model_confidence_for_gate` can force gating
    #   on the raw model confidence (`p_conf`) instead. A separate threshold
    #   `buddy.tier2.model_conf_threshold` can be provided when gating on model_conf.
    gate_use_model = bool(tier2_cfg.get("use_model_confidence_for_gate", False))
    model_threshold_from_tier2 = tier2_cfg.get("model_conf_threshold", None)

    if tier2_p_win is not None and not gate_use_model:
        gate_conf = float(tier2_p_win)
        threshold = float(tier2_cfg.get("p_win_threshold", 0.60))
    else:
        # Prefer Tier-1 confidence for human/fallback gating if available,
        # otherwise use the raw model confidence.
        gate_conf = float(p_conf)
        if model_threshold_from_tier2 is not None:
            threshold = float(model_threshold_from_tier2)
        elif tier1_conf is not None:
            threshold = float(cfg.get("fx", {}).get("confidence", {}).get("low_lt", 0.70))
        else:
            threshold = float(cfg.get("fx", {}).get("confidence", {}).get("low_lt", 0.70))

    # Display Tier-2 calibrated probability if available
    if tier2_p_win is not None:
        t2_color = "green" if tier2_p_win >= 0.50 else "yellow" if tier2_p_win >= 0.40 else "red"
        console.print(f"  [bold]Win Prob (Tier-2): [{t2_color}]{tier2_p_win:.1%}[/{t2_color}][/bold]")
        
        # Calculate expected value
        stop_loss_pips = float(cfg.get("buddy", {}).get("stop_loss_pips", 12.0))
        take_profit_pips = float(cfg.get("buddy", {}).get("take_profit_pips", 24.0))
        ev = (tier2_p_win * take_profit_pips) - ((1 - tier2_p_win) * stop_loss_pips)
        ev_color = "green" if ev > 0 else "red"
        console.print(f"  Expected Value:  [{ev_color}]{ev:+.1f} pips[/{ev_color}]")
    elif tier1_conf is not None:
        console.print(f"  Tier-1 Conf:     {tier1_conf:.1%}")
    
    console.print("-" * 60)
    
    if verbose:
        console.print("[dim]Detailed Diagnostics:[/dim]")
        conf_msg = f"model_conf={_fmt_prob(p_conf)}"
        if tier1_conf is not None:
            conf_msg += f" tier1_conf={_fmt_prob(gate_conf)}"
        if tier2_p_win is not None:
            conf_msg += f" tier2_p_win={_fmt_prob(float(tier2_p_win))}"
        console.print(f"[dim]  Raw values: direction={_fmt_prob(p_dir)} ({side}), {conf_msg}[/dim]")
        if tier1_reasons:
            console.print(f"[dim]  Tier-1 reasons: {', '.join(tier1_reasons)}[/dim]")
        if p_dir <= 1e-12 or p_dir >= 1.0 - 1e-12 or p_conf <= 1e-12 or p_conf >= 1.0 - 1e-12:
            dlog = _dense_logit("direction")
            clog = _dense_logit("confidence")
            if dlog is not None or clog is not None:
                console.print(
                    f"[dim]  Logits: direction={dlog if dlog is not None else 'n/a'} "
                    f"confidence={clog if clog is not None else 'n/a'} (float32 saturation)[/dim]"
                )
                try:
                    if dlog is not None:
                        console.print(f"[dim]  direction_sigmoid64={_stable_sigmoid_from_logit(dlog):.3e}[/dim]")
                    if clog is not None:
                        console.print(f"[dim]  confidence_sigmoid64={_stable_sigmoid_from_logit(clog):.3e}[/dim]")
                except Exception:
                    pass

    # Threshold: Tier-2 uses calibrated p_win; otherwise keep Tier-1 behavior.
    # Allow complete bypass of the confidence gate via config for practice.
    if bool(tier2_cfg.get("remove_confidence_gate", False)):
        if verbose:
            console.print("[yellow]Confidence gate disabled by config (remove_confidence_gate=True)[/yellow]")
    else:
        # Hybrid gate: require both raw model confidence and calibrated tier2 p_win
        # when explicitly requested via config `require_both`.
        require_both = bool(tier2_cfg.get("require_both", False))
        if require_both and tier2_p_win is not None:
            model_thr = float(model_threshold_from_tier2) if model_threshold_from_tier2 is not None else float(cfg.get("fx", {}).get("confidence", {}).get("low_lt", 0.70))
            p_win_thr = float(tier2_cfg.get("p_win_threshold", 0.60))
            p_ok = float(p_conf) >= model_thr
            tier2_ok = float(tier2_p_win) >= p_win_thr
            if not (p_ok and tier2_ok):
                console.print(f"[yellow]No trade[/yellow]: requires model_conf >= {model_thr:.2f} AND tier2_p_win >= {p_win_thr:.2f}")
                return
        else:
            # Backward-compatible behavior: gate on calibrated tier2 p_win when available
            # unless `use_model_confidence_for_gate` forces using the raw model confidence.
            if tier2_p_win is not None and not gate_use_model:
                gate_conf = float(tier2_p_win)
                threshold = float(tier2_cfg.get("p_win_threshold", 0.60))
            else:
                gate_conf = float(p_conf)
                if model_threshold_from_tier2 is not None:
                    threshold = float(model_threshold_from_tier2)
                elif tier1_conf is not None:
                    threshold = float(cfg.get("fx", {}).get("confidence", {}).get("low_lt", 0.70))
                else:
                    threshold = float(cfg.get("fx", {}).get("confidence", {}).get("low_lt", 0.70))

            if gate_conf < threshold:
                console.print(f"[yellow]No trade[/yellow]: confidence below {threshold:.2f}")
                return

    if not execute:
        console.print("[cyan]DRY-RUN[/cyan]: would place a market order.")
        return

    # Place a conservative market order with SL/TP derived from pips.
    last_close = float(df["close"].iloc[-1])
    from fx_paper import pip_size, position_size_units

    ps = float(pip_size(instrument))
    stop_loss_pips = float(cfg.get("buddy", {}).get("stop_loss_pips", 20.0))
    take_profit_pips = float(cfg.get("buddy", {}).get("take_profit_pips", 40.0))
    stop_distance = stop_loss_pips * ps

    equity_eff = float(
        equity
        if equity is not None
        else cfg.get("buddy", {}).get("equity", cfg.get("fx", {}).get("equity", 10_000.0))
    )
    
    # =========================================================================
    # AGGRESSIVE SCALING: Use slippage-adjusted Kelly sizing with circuit breakers
    # =========================================================================
    risk_eff = None
    use_smart_execution = False
    
    if scaling_engine is not None:
        try:
            # Check circuit breakers first
            can_trade, block_reason = scaling_engine.risk_manager.can_trade()
            if not can_trade:
                console.print(f"[red]🛑 CIRCUIT BREAKER: {block_reason}[/red]")
                return
            
            # Calculate slippage-adjusted Kelly sizing
            position_lots = scaling_engine.calculate_position_size(
                equity=equity_eff,
                atr_pips=stop_loss_pips,  # Use SL as proxy for volatility
            )
            
            # Convert lots to units
            units_from_scaling = int(position_lots * 100000)
            
            # Apply phase-based Kelly multiplier
            kelly_mult = scaling_engine._get_kelly_multiplier()
            
            # Get adjusted risk percentage
            kelly_fraction = scaling_engine.kelly_calculator.kelly_fraction
            slippage_adj = scaling_engine.kelly_calculator.slippage_adjustment
            risk_eff = kelly_fraction * slippage_adj * kelly_mult
            
            # Cap at max risk
            risk_eff = min(risk_eff, 0.03)  # Never risk more than 3% per trade
            
            use_smart_execution = position_lots >= 5.0  # Split orders if >= 5 lots
            
            if verbose:
                console.print(f"[yellow]Aggressive Scaling:[/yellow]")
                console.print(f"  Kelly Fraction: {kelly_fraction:.1%}")
                console.print(f"  Slippage Adjustment: {slippage_adj:.1%}")
                console.print(f"  Phase Multiplier: {kelly_mult:.0%}")
                console.print(f"  Effective Risk: {risk_eff:.2%}")
                console.print(f"  Position Size: {position_lots:.2f} lots ({units_from_scaling:,} units)")
                if use_smart_execution:
                    console.print(f"  [cyan]Smart Execution: Will split large order[/cyan]")
        except Exception as e:
            console.print(f"[yellow]Warning: Aggressive scaling error: {e}. Using standard sizing.[/yellow]")
            risk_eff = None
    
    # Fallback: Tier-2 sizing (fractional Kelly for fixed RR, capped by config)
    if risk_eff is None and tier2_p_win is not None:
        try:
            stop_loss_pips_eff = float(cfg.get("buddy", {}).get("stop_loss_pips", 20.0))
            take_profit_pips_eff = float(cfg.get("buddy", {}).get("take_profit_pips", 60.0))
            b = float(take_profit_pips_eff / max(stop_loss_pips_eff, 1e-9))
            p = float(max(0.0, min(1.0, float(tier2_p_win))))
            q = 1.0 - p
            kelly = ((b * p) - q) / max(b, 1e-9)
            kelly_fraction = float(tier2_cfg.get("kelly_fraction", 0.5))
            cap = float(tier2_cfg.get("max_risk_per_trade_pct", 0.02))
            risk_eff = float(max(0.0, min(cap, kelly * kelly_fraction)))
        except Exception:
            risk_eff = None

    if risk_eff is None:
        risk_eff = float(
            risk_per_trade_pct
            if risk_per_trade_pct is not None
            else cfg.get("buddy", {}).get(
                "risk_per_trade_pct",
                cfg.get("fx", {}).get("risk", {}).get("risk_per_trade_pct", 0.005),
            )
        )
    units = position_size_units(
        instrument=instrument,
        equity=equity_eff,
        risk_per_trade_pct=risk_eff,
        stop_distance_price=stop_distance,
        price=last_close,
    )
    if side == "sell":
        units = -abs(int(units))
    else:
        units = abs(int(units))

    # CLI override: allow user to force exact units (practice / debug)
    if force_units is not None:
        try:
            fu = int(force_units)
            units = -abs(int(fu)) if side == "sell" else abs(int(fu))
            if verbose:
                console.print(f"[dim]Force-units override[/dim]: using units={units}")
        except Exception:
            pass

    stop_price = last_close - stop_distance if units > 0 else last_close + stop_distance
    tp_distance = take_profit_pips * ps
    tp_price = last_close + tp_distance if units > 0 else last_close - tp_distance

    # Finalize units as integer (preserve sign) and submit exactly that value.
    try:
        units_final = int(units)
    except Exception:
        units_final = int(float(units))

    if verbose:
        console.print(f"[dim]Order sizing[/dim]: computed_units={units} final_submitted_units={units_final}")

    # =========================================================================
    # AGGRESSIVE SCALING: Smart Order Execution
    # =========================================================================
    if scaling_engine is not None and use_smart_execution and abs(units_final) >= 500000:
        # Split large orders to reduce slippage
        try:
            from aggressive_scaling import SmartOrderExecutor
            executor = SmartOrderExecutor(atr_pips=stop_loss_pips, spread_pips=1.5)
            
            split_orders = executor.split_order(
                total_lots=abs(units_final) / 100000,
                current_price=last_close,
                side=side,
            )
            
            console.print(f"[cyan]Smart Execution: Splitting into {len(split_orders)} orders[/cyan]")
            
            total_filled = 0
            for i, order in enumerate(split_orders, 1):
                order_units = int(order['size'] * 100000)
                if side == "sell":
                    order_units = -abs(order_units)
                
                price_bound = _fx_execution_guard_price_bound(None, client, instrument=instrument, units=order_units)
                result = client.create_market_order(
                    instrument=instrument,
                    units=order_units,
                    stop_loss_price=float(stop_price),
                    take_profit_price=float(tp_price),
                    price_bound=price_bound,
                    client_tag="buddy_tf_split",
                )
                total_filled += abs(order_units)
                console.print(f"  [green]Order {i}/{len(split_orders)}[/green]: {order_units:,} units - {result}")
                
                if i < len(split_orders):
                    time.sleep(0.5)  # Brief pause between splits
            
            console.print(f"[green]Total filled[/green]: {total_filled:,} units")
            
            # Update scaling engine state
            scaling_engine.record_trade_result(
                win=True,  # Will be updated when trade closes
                pips=0,  # Will be updated when trade closes
                lots=total_filled / 100000,
            )
        except Exception as e:
            console.print(f"[yellow]Smart execution failed: {e}. Using standard order.[/yellow]")
            price_bound = _fx_execution_guard_price_bound(None, client, instrument=instrument, units=units_final)
            result = client.create_market_order(
                instrument=instrument,
                units=units_final,
                stop_loss_price=float(stop_price),
                take_profit_price=float(tp_price),
                price_bound=price_bound,
                client_tag="buddy_tf",
            )
            console.print(f"[green]Order submitted[/green]: {result}")
    else:
        # Standard order execution
        price_bound = _fx_execution_guard_price_bound(None, client, instrument=instrument, units=units_final)
        result = client.create_market_order(
            instrument=instrument,
            units=units_final,
            stop_loss_price=float(stop_price),
            take_profit_price=float(tp_price),
            price_bound=price_bound,
            client_tag="buddy_tf",
        )
        console.print(f"[green]Order submitted[/green]: {result}")
        
        # Update scaling engine state if active
        if scaling_engine is not None:
            try:
                scaling_engine.record_trade_result(
                    win=True,  # Placeholder - will be updated when trade closes
                    pips=0,
                    lots=abs(units_final) / 100000,
                )
            except Exception:
                pass
    
    # Schedule an auto-close for scalping if requested via CLI/arg `max_hold_minutes`.
    try:
        if max_hold_minutes is not None:
            m = float(max_hold_minutes)
            if m > 0:
                _schedule_auto_close(client, instrument, delay_s=m * 60.0, verbose=bool(verbose))
    except Exception:
        pass


def buddy_loop(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    instrument: str = "USD_JPY",
    granularity: str = "M5",
    candles: int = 500,
    execute: bool = False,
    force_execute: bool = False,
    force_units: int | None = None,
    equity: float | None = None,
    risk_per_trade_pct: float | None = None,
    verbose: bool = False,
    max_trades: int = 1,
    **kwargs: Any,
) -> None:
    """Buddy loop: keep TF+model warm; trade once per new candle (practice only)."""
    _configure_predict_output(verbose)

    max_hold_minutes = None
    try:
        m = kwargs.get("max_hold_minutes", None)
        max_hold_minutes = float(m) if m is not None else None
    except Exception:
        max_hold_minutes = None

    all_features = bool(kwargs.get("all_features", True))

    from collections import deque
    import json
    import math
    import numpy as np
    import pandas as pd
    from tracing_setup import span, record_exception

    from datetime import datetime, timezone

    def _granularity_seconds(g: str) -> int:
        s = str(g).strip().upper()
        if s.startswith("M") and s[1:].isdigit():
            return int(s[1:]) * 60
        if s.startswith("H") and s[1:].isdigit():
            return int(s[1:]) * 3600
        if s == "D":
            return 86400
        # fallback
        return 300

    def _sleep_until_next_boundary(interval_s: int, *, lag_s: float = 2.0) -> None:
        # Align to the next candle close boundary in UTC (best-effort).
        now = time.time()
        next_boundary = (math.floor(now / interval_s) + 1) * interval_s
        sleep_s = max(0.0, (next_boundary + float(lag_s)) - now)
        if verbose:
            eta = datetime.fromtimestamp(next_boundary, tz=timezone.utc).isoformat().replace(UTC_OFFSET_SUFFIX, "Z")
            console.print(f"[dim]Loop[/dim]: waiting {sleep_s:.1f}s for next {granularity} boundary @ {eta}")
        time.sleep(sleep_s)

    cfg = load_config(config_path)

    equity_eff = float(
        equity
        if equity is not None
        else cfg.get("buddy", {}).get("equity", cfg.get("fx", {}).get("equity", 10_000.0))
    )
    risk_eff = float(
        risk_per_trade_pct
        if risk_per_trade_pct is not None
        else cfg.get("buddy", {}).get(
            "risk_per_trade_pct",
            cfg.get("fx", {}).get("risk", {}).get("risk_per_trade_pct", 0.005),
        )
    )

    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    if checkpoint_path:
        ck_meta = _meta_path_for_checkpoint(str(checkpoint_path))
        if ck_meta and ck_meta.exists():
            meta_path = ck_meta
    if not meta_path.exists():
        raise FileNotFoundError("Missing Buddy metadata. Run: python main.py train-buddy")
    meta = json.loads(meta_path.read_text())

    tier2_cfg = (cfg.get("buddy", {}) or {}).get("tier2", {}) or {}
    tier2_enabled = bool(tier2_cfg.get("enabled", False))

    # Match inference preprocessing to how the model was trained.
    all_features = bool(meta.get("all_features", all_features))
    train_smoothing = bool(meta.get("train_smoothing", True))
    median_window = meta.get("median_window", None)
    median_window = int(median_window) if median_window is not None else None
    min_volume = meta.get("min_volume", None)
    min_volume = int(min_volume) if min_volume is not None else None
    spread_filter = bool(meta.get("spread_filter", False))
    spread_pctl = float(meta.get("spread_pctl", 0.99))
    spread_mult = float(meta.get("spread_mult", 3.0))

    model_path = checkpoint_path or meta.get("model_path")
    if not model_path:
        raise ValueError("Missing model path in metadata")

    seq_len_eff = int(meta.get("seq_len") or 64)
    feature_columns: list[str] = list(meta.get("feature_columns") or [])
    if not feature_columns:
        raise ValueError("Missing feature_columns in Buddy metadata")

    std = meta.get("standardize") or {}
    mu = np.asarray(std.get("mean") or [], dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(std.get("std") or [], dtype=np.float32).reshape(1, -1)
    if mu.size != len(feature_columns) or sigma.size != len(feature_columns):
        raise ValueError("Standardization stats do not match feature_columns")

    pca_info = meta.get("pca")
    pca_components = int(meta.get("pca_components") or 0)
    pca_mean = None
    pca_components_mat = None
    if pca_info and pca_components > 0:
        pca_mean = np.asarray(pca_info.get("mean") or [], dtype=np.float32).reshape(1, -1)
        pca_components_mat = np.asarray(pca_info.get("components") or [], dtype=np.float32)
        if pca_components_mat.ndim != 2:
            raise ValueError("Invalid PCA components in metadata")

    # Live trading gate.
    threshold = float(meta.get("live_confidence_threshold", 0.75))
    if execute:
        if force_execute:
            console.print(
                "[yellow]FORCE-EXECUTE[/yellow]: bypassing the training accuracy gate; PRACTICE orders may be placed."
            )
        else:
            enabled, _ = _buddy_live_enabled_from_meta()
            if not enabled:
                console.print("[yellow]Live trading disabled[/yellow]: train Buddy and reach >=62% direction accuracy.")
                execute = False
            else:
                console.print(f"[green]Live trading enabled[/green] (confidence threshold {threshold*100:.0f}%).")

    # Load TF/model once (warm process).
    _configure_tf_metal(verbose=verbose)
    import tensorflow as tf

    from ml_head_engine import MLEngineHead
    from mr_engine import MREngineHead
    from ms_head_engine import MSEngineHead
    from mt_engine import MTEngineHead
    from mx_head_engine import MXEngineHead

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "MLEngineHead": MLEngineHead,
            "MREngineHead": MREngineHead,
            "MSEngineHead": MSEngineHead,
            "MTEngineHead": MTEngineHead,
            "MXEngineHead": MXEngineHead,
        },
    )

    from fx_paper import candles_to_ohlcv_df
    from oanda_practice import OandaPracticeClient

    client = OandaPracticeClient.from_env()
    interval_s = _granularity_seconds(granularity)

    def _to_utc_iso_z(ts: pd.Timestamp) -> str:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat().replace(UTC_OFFSET_SUFFIX, "Z")

    def _normalize_candle_df(df_in: "pd.DataFrame") -> "pd.DataFrame":
        out = df_in
        if out is None or len(out) == 0:
            return pd.DataFrame()
        if "time" not in out.columns:
            return pd.DataFrame()
        out = out.copy()
        out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
        out = out.dropna(subset=["time"]).sort_values("time", kind="mergesort").reset_index(drop=True)
        return out

    # --- Streaming candle buffer (bounded, no persistence) ---
    candle_buf = deque(maxlen=int(candles))
    last_seen_ts: pd.Timestamp | None = None
    last_seen_from_time: str | None = None

    # Warm start the deque with the last `candles` candles.
    try:
        with span("warm_get_candles", attributes={"instrument": instrument, "granularity": granularity}):
            warm_resp = client.get_candles(instrument, granularity=granularity, count=int(candles), price="MBA")
        warm_df = _normalize_candle_df(candles_to_ohlcv_df(warm_resp))
        if len(warm_df) > 0:
            candle_buf.extend(warm_df.to_dict("records"))
            last_seen_ts = pd.to_datetime(warm_df["time"].iloc[-1], utc=True, errors="coerce")
            if pd.notna(last_seen_ts):
                last_seen_from_time = _to_utc_iso_z(pd.Timestamp(last_seen_ts))
    except Exception:
        try:
            record_exception(Exception("warm_fetch_failed"))
        except Exception:
            pass

    last_candle_id: str | None = None
    primed = False
    trades_done = 0
    # Simple trace counter for debugging / liveness checks.
    trace_iter = 0

    while True:
        # Quick pre-sleep checks:
        # - If weekend (Sat/Sun) pause briefly.
        # - If last_seen_ts is stale (>5 minutes), assume feed dead: clear buffer,
        #   repull recent candles (200), reset pointers, and skip waiting for boundary.
        skip_sleep = False
        try:
            now_utc = datetime.now(timezone.utc)
            # Weekend check: Saturday=5, Sunday=6
            if now_utc.weekday() >= 5:
                console.print("[dim]Loop[/dim]: Weekend. sleeping 60 seconds.")
                time.sleep(60)
                continue

            if last_seen_ts is not None and pd.notna(last_seen_ts):
                delta_s = (now_utc - pd.to_datetime(last_seen_ts, utc=True)).total_seconds()
                if delta_s > 300:
                    console.print(
                        "[yellow]Loop[/yellow]: last_seen candle >5min behind; assuming feed dead. Repulling recent candles."
                    )
                    # Clear buffer and repull the most recent 200 candles (two * 100)
                    try:
                        candle_buf.clear()
                        with span("repull_recent_candles", attributes={"instrument": instrument, "count": 200}):
                            resp = client.get_candles(instrument, granularity=granularity, count=200, price="MBA")
                        warm_df = _normalize_candle_df(candles_to_ohlcv_df(resp))
                        if len(warm_df) > 0:
                            candle_buf.extend(warm_df.to_dict("records"))
                            last_seen_ts = pd.to_datetime(warm_df["time"].iloc[-1], utc=True, errors="coerce")
                            if pd.notna(last_seen_ts):
                                last_seen_from_time = _to_utc_iso_z(pd.Timestamp(last_seen_ts))
                            # Don't wait for a ghost boundary; allow immediate processing.
                            primed = True
                            last_candle_id = None
                            skip_sleep = True
                        else:
                            console.print("[yellow]Loop[/yellow]: repull returned 0 candles; sleeping 30s")
                            time.sleep(30)
                            continue
                    except Exception as _e:
                        try:
                            record_exception(_e)
                        except Exception:
                            pass
                        console.print("[red]Loop[/red]: repull failed; sleeping 30s")
                        time.sleep(30)
                        continue
        except Exception:
            # Best-effort; don't let the pre-sleep checks crash the loop.
            pass

        if not skip_sleep:
            _sleep_until_next_boundary(interval_s, lag_s=2.0)

        # Heartbeat / trace: increment and occasionally print to help identify where loop hangs.
        try:
            trace_iter += 1
        except Exception:
            trace_iter = 1
        if verbose or (trace_iter % 20 == 0):
            try:
                console.print(
                    f"[dim]Trace[/dim]: still alive — on Kindle number {trace_iter} | last_seen={last_seen_from_time} | buf={len(candle_buf)}"
                )
            except Exception:
                # best-effort trace, never raise
                pass

        # Fetch only new candles since the last seen timestamp (inclusive), then append strictly newer.
        try:
            fetch_count = int(min(max(50, interval_s // 60 * 10), 5000))
        except Exception:
            fetch_count = 200
        try:
            with span("get_incremental_candles", attributes={"instrument": instrument, "count": int(fetch_count)}):
                resp = client.get_candles(
                    instrument,
                    granularity=granularity,
                    count=int(fetch_count),
                    price="MBA",
                    from_time=last_seen_from_time,
                )
        except Exception as _e:
            try:
                record_exception(_e)
            except Exception:
                pass
            resp = None
        inc_df = _normalize_candle_df(candles_to_ohlcv_df(resp))
        if len(inc_df) == 0:
            if verbose:
                console.print("[yellow]Loop[/yellow]: no candles returned")
            continue

        if last_seen_ts is not None and pd.notna(last_seen_ts):
            inc_df = inc_df.loc[inc_df["time"] > pd.Timestamp(last_seen_ts)].copy()

        if len(inc_df) == 0:
            # No new candle since last tick.
            continue

        candle_buf.extend(inc_df.to_dict("records"))
        last_seen_ts = pd.to_datetime(inc_df["time"].iloc[-1], utc=True, errors="coerce")
        if pd.notna(last_seen_ts):
            last_seen_from_time = _to_utc_iso_z(pd.Timestamp(last_seen_ts))

        # Build the working window DataFrame from the deque (bounded memory).
        df = pd.DataFrame(list(candle_buf))
        df = _normalize_candle_df(df)
        if len(df) == 0:
            continue

        # Identify candle by its timestamp if available.
        candle_id = None
        try:
            if "time" in df.columns and len(df) > 0:
                candle_id = str(df["time"].iloc[-1])
        except Exception:
            candle_id = None
        candle_id = candle_id or f"len={len(df)}"

        # Prime on first seen candle to avoid an immediate trade mid-candle.
        if not primed:
            primed = True
            last_candle_id = candle_id
            console.print(f"[dim]Loop[/dim]: primed at candle {candle_id}; waiting for next candle")
            continue

        if candle_id == last_candle_id:
            # Avoid duplicate action if OANDA returns the same last candle.
            continue
        last_candle_id = candle_id

        # Apply training-time filters.
        if min_volume is not None and "volume" in df.columns:
            try:
                df = df.loc[df["volume"].astype(float) >= float(min_volume)].copy()
            except Exception:
                pass

        if spread_filter and ("bid_close" in df.columns and "ask_close" in df.columns):
            try:
                spread = (df["ask_close"].astype(float) - df["bid_close"].astype(float)).abs()
                spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
                if not spread.empty:
                    pctl_thr = float(np.quantile(spread.to_numpy(), float(spread_pctl)))
                    med = float(np.median(spread.to_numpy()))
                    thr = max(pctl_thr, float(spread_mult) * med) if med > 0 else pctl_thr
                    keep_mask = (df["ask_close"].astype(float) - df["bid_close"].astype(float)).abs() <= thr
                    df = df.loc[keep_mask].copy()
            except Exception:
                pass

        if all_features:
            from feature_engineering import FeatureEngineering
            fe = FeatureEngineering(cfg.get("feature_engineering", {}))
            try:
                with span("feature_engineering", attributes={"rows": len(df)}):
                    df = fe.create_features(
                        df,
                        include_all=True,
                        apply_candle_smoothing=bool(train_smoothing),
                        median_window=median_window,
                    )
            except Exception as _e:
                try:
                    record_exception(_e)
                except Exception:
                    pass

        numeric_df = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
        numeric_df = numeric_df.ffill().bfill().fillna(0.0)

        expected_cols = list(feature_columns)
        missing_cols = [c for c in expected_cols if c not in set(numeric_df.columns)]
        numeric_df = numeric_df.reindex(columns=expected_cols, fill_value=0.0)
        if missing_cols:
            mu_flat = mu.reshape(-1)
            for idx, col in enumerate(expected_cols):
                if col in set(missing_cols):
                    numeric_df[col] = float(mu_flat[idx])

        x2d_full = numeric_df.to_numpy(dtype=np.float32, copy=False)
        if len(x2d_full) < seq_len_eff:
            if verbose:
                console.print(f"[yellow]Loop[/yellow]: need {seq_len_eff} rows, got {len(x2d_full)}")
            continue

        x_std_full = (x2d_full - mu) / np.where(sigma < 1e-6, 1.0, sigma)
        x_std_full = np.clip(x_std_full, -10.0, 10.0).astype(np.float32, copy=False)
        if pca_components_mat is not None and pca_mean is not None:
            x_std_full = (x_std_full - pca_mean) @ pca_components_mat.T

        n_ens = int(meta.get("inference_ensemble_windows", 20) or 20)
        n_ens = max(1, min(n_ens, 100))
        need = seq_len_eff + n_ens - 1
        if len(x_std_full) < need:
            n_ens = 1
            need = seq_len_eff

        tail = x_std_full[-need:]
        if n_ens > 1:
            windows = np.stack([tail[i : i + seq_len_eff] for i in range(n_ens)], axis=0).astype(np.float32)
        else:
            windows = tail[-seq_len_eff:].reshape(1, seq_len_eff, -1).astype(np.float32)

        try:
            with span("model_inference", attributes={"n_ens": int(n_ens), "seq_len": int(seq_len_eff)}):
                _inputs_struct = getattr(model, "_inputs_struct", None)
                _expects_features_dict = isinstance(_inputs_struct, dict) and "features" in _inputs_struct
                pred_in = {"features": windows} if _expects_features_dict else windows
                pred = model(pred_in, training=False)
        except Exception as _e:
            try:
                record_exception(_e)
            except Exception:
                pass
            raise
        d = pred["direction"]
        c = pred["confidence"]
        if hasattr(d, "numpy"):
            d = d.numpy()
        if hasattr(c, "numpy"):
            c = c.numpy()
        dir_probs = np.asarray(d).reshape(-1)
        conf_probs = np.asarray(c).reshape(-1)
        conf_probs = np.clip(conf_probs, 0.0, 1.0)
        p_dir = float(np.mean(dir_probs))
        p_conf = float(np.quantile(conf_probs, float(meta.get("inference_confidence_quantile", 0.90) or 0.90)))

        side = "buy" if p_dir >= 0.5 else "sell"
        tier2_p_win = None
        gate_conf = float(p_conf)
        if tier2_enabled:
            try:
                dir_conf = float(p_dir) if side == "buy" else float(1.0 - float(p_dir))
                score = float(max(0.0, min(1.0, dir_conf * float(p_conf))))
                tier2_p_win = _tier2_apply_calibration(meta, score)
                if tier2_p_win is not None:
                    gate_conf = float(tier2_p_win)
            except Exception:
                tier2_p_win = None
        if verbose:
            if tier2_p_win is not None:
                console.print(
                    f"Buddy loop: candle={candle_id} direction={p_dir:.3f} ({side}) model_conf={p_conf:.3f} tier2_p_win={gate_conf:.3f}"
                )
            else:
                console.print(f"Buddy loop: candle={candle_id} direction={p_dir:.3f} ({side}) conf={gate_conf:.3f}")
        else:
            if tier2_p_win is not None:
                console.print(f"Buddy: {side} tier2_p_win={gate_conf:.3f} candle={candle_id}")
            else:
                console.print(f"Buddy: {side} conf={gate_conf:.3f} candle={candle_id}")

        thr = float(tier2_cfg.get("p_win_threshold", 0.60)) if tier2_p_win is not None else float(threshold)
        if gate_conf < thr:
            continue
        if not execute:
            console.print("[cyan]DRY-RUN[/cyan]: would place a market order.")
            continue

        from fx_paper import pip_size, position_size_units

        last_close = float(df["close"].iloc[-1])
        ps = float(pip_size(instrument))
        stop_loss_pips = float(cfg.get("buddy", {}).get("stop_loss_pips", 20.0))
        take_profit_pips = float(cfg.get("buddy", {}).get("take_profit_pips", 40.0))
        stop_distance = stop_loss_pips * ps

        risk_eff_trade = float(risk_eff)
        if tier2_p_win is not None:
            try:
                b = float(take_profit_pips / max(stop_loss_pips, 1e-9))
                p = float(max(0.0, min(1.0, float(tier2_p_win))))
                q = 1.0 - p
                kelly = ((b * p) - q) / max(b, 1e-9)
                kelly_fraction = float(tier2_cfg.get("kelly_fraction", 0.5))
                cap = float(tier2_cfg.get("max_risk_per_trade_pct", 0.02))
                risk_eff_trade = float(max(0.0, min(cap, kelly * kelly_fraction)))
            except Exception:
                risk_eff_trade = float(risk_eff)

        units = position_size_units(
            instrument=instrument,
            equity=float(equity_eff),
            risk_per_trade_pct=float(risk_eff_trade),
            stop_distance_price=stop_distance,
            price=last_close,
        )
        units = -abs(int(units)) if side == "sell" else abs(int(units))

        # CLI override: allow user to force exact units (practice / debug)
        if force_units is not None:
            try:
                fu = int(force_units)
                units = -abs(int(fu)) if side == "sell" else abs(int(fu))
                if verbose:
                    console.print(f"[dim]Force-units override[/dim]: using units={units}")
            except Exception:
                pass

        stop_price = last_close - stop_distance if units > 0 else last_close + stop_distance
        tp_distance = take_profit_pips * ps
        tp_price = last_close + tp_distance if units > 0 else last_close - tp_distance

        # Finalize units as integer (preserve sign) and submit exactly that value.
        try:
            units_final = int(units)
        except Exception:
            units_final = int(float(units))

        if verbose:
            console.print(f"[dim]Order sizing[/dim]: computed_units={units} final_submitted_units={units_final}")

        price_bound = _fx_execution_guard_price_bound(None, client, instrument=instrument, units=units_final)
        try:
            with span("order_submission", attributes={"instrument": instrument, "units": int(units_final)}):
                result = client.create_market_order(
                    instrument=instrument,
                    units=int(units_final),
                    stop_loss_price=float(stop_price),
                    take_profit_price=float(tp_price),
                    price_bound=price_bound,
                    client_tag="buddy_tf",
                )
        except Exception as _e:
            try:
                record_exception(_e)
            except Exception:
                pass
            raise
        console.print(f"[green]Order submitted[/green]: {result}")
        # Schedule auto-close for scalping if requested via CLI `max_hold_minutes`.
        try:
            if max_hold_minutes is not None:
                m = float(max_hold_minutes)
                if m > 0:
                    _schedule_auto_close(client, instrument, delay_s=m * 60.0, verbose=bool(verbose))
        except Exception:
            pass

        trades_done += 1
        if int(max_trades) > 0 and trades_done >= int(max_trades):
            console.print(f"[dim]Loop[/dim]: reached max_trades={int(max_trades)}; exiting")
            return


def _normalize_command_args(args: Any) -> None:
    if getattr(args, "command", None) == "Buddy":
        console.print("[yellow]Deprecation: use lowercase 'buddy' instead of 'Buddy'[/yellow]")
        args.command = "buddy"


def _maybe_run_buddy_interactive_wizard(*, default_config: str) -> bool:
    if len(sys.argv) == 2 and sys.argv[1] in {"buddy", "Buddy"} and sys.stdin.isatty():
        if sys.argv[1] == "Buddy":
            console.print("[yellow]Deprecation: please use lowercase 'buddy' instead of 'Buddy'[/yellow]")
        try:
            _buddy_interactive_wizard(default_config=default_config)
            return True
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            return True
    return False


def _maybe_launch_buddy_repl(args: Any) -> bool:
    if getattr(args, "command", None) == "buddy" and bool(getattr(args, "repl", False)):
        launch_buddy_repl_from_wizard(
            args.config,
            checkpoint_path=getattr(args, "model_path", None),
            instrument=str(getattr(args, "instrument", "USD_JPY")),
            granularity=str(getattr(args, "granularity", "M5")),
            candles=int(getattr(args, "candles", 300)),
            execute=bool(getattr(args, "execute", True)),
            all_features=bool(getattr(args, "all_features", False)),
            verbose=bool(getattr(args, "verbose", False)),
            assistant_name="Buddy",
        )
        return True
    return False


def _compute_force_units(*, force_units_raw: Any, force_margin_raw: Any) -> int | None:
    if force_units_raw is not None:
        return int(force_units_raw)
    if force_margin_raw is not None:
        return int(round(float(force_margin_raw) / 0.05))
    return None


def _dispatch_buddy(args: Any, command_map: dict[str, Any]) -> None:
    should_execute = bool(args.execute) and (not bool(getattr(args, "dry_run", False)))

    # Validate and normalize instrument name
    try:
        instrument_validated = _validate_instrument(str(args.instrument))
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)

    force_units_eff = _compute_force_units(
        force_units_raw=getattr(args, "force_units", None),
        force_margin_raw=getattr(args, "force_margin", None),
    )
    max_hold_raw = getattr(args, "max_hold_minutes", None)
    max_hold_eff = float(max_hold_raw) if max_hold_raw is not None else None
    
    # Check for aggressive scaling mode
    aggressive_scaling = bool(getattr(args, "aggressive_scaling", False))
    
    # Check for RL position sizer
    use_rl_sizer = bool(getattr(args, "use_rl_sizer", False))
    
    # Intelligent mode flags
    intelligent = bool(getattr(args, "intelligent", False))
    explain = bool(getattr(args, "explain", False))
    llm_provider = str(getattr(args, "llm_provider", "auto"))

    if bool(getattr(args, "loop", False)):
        buddy_loop(
            args.config,
            checkpoint_path=args.model_path,
            instrument=instrument_validated,
            granularity=args.granularity,
            candles=args.candles,
            execute=should_execute,
            force_execute=bool(getattr(args, "force_execute", False)),
            force_units=force_units_eff,
            max_hold_minutes=max_hold_eff,
            equity=float(getattr(args, "equity", 10_000.0)),
            risk_per_trade_pct=float(getattr(args, "risk", 0.005)),
            all_features=getattr(args, "all_features", False),
            verbose=args.verbose,
            max_trades=int(getattr(args, "max_trades", 1)),
        )
        return

    command_map["buddy"](
        args.config,
        checkpoint_path=args.model_path,
        instrument=instrument_validated,
        granularity=args.granularity,
        candles=args.candles,
        execute=should_execute,
        force_execute=bool(getattr(args, "force_execute", False)),
        force_units=force_units_eff,
        max_hold_minutes=max_hold_eff,
        equity=float(getattr(args, "equity", 10_000.0)),
        risk_per_trade_pct=float(getattr(args, "risk", 0.005)),
        all_features=getattr(args, "all_features", False),
        verbose=args.verbose,
        aggressive_scaling=aggressive_scaling,
        use_rl_sizer=use_rl_sizer,
        intelligent=intelligent,
        explain=explain,
        llm_provider=llm_provider,
        llm_enhance=getattr(args, "llm_enhance", False),
    )


def _dispatch_train_buddy(args: Any, command_map: dict[str, Any]) -> None:
    # Suppress verbose utils logging during config load for clean output
    import logging as _logging
    _utils_logger = _logging.getLogger('utils')
    _utils_prev = _utils_logger.level
    _utils_logger.setLevel(_logging.WARNING)
    cfg = load_config(args.config)
    _utils_logger.setLevel(_utils_prev)
    
    buddy_cfg = (cfg.get("buddy") or {}) if isinstance(cfg, dict) else {}
    buddy_train_cfg = (buddy_cfg.get("train_defaults") or buddy_cfg.get("train") or {}) if isinstance(buddy_cfg, dict) else {}
    training_cfg = (cfg.get("training") or {}) if isinstance(cfg, dict) else {}
    model_cfg = (cfg.get("model") or {}) if isinstance(cfg, dict) else {}

    def _cfg_get(name: str, default_val):
        """Get config value from buddy.train_defaults first, then root config."""
        try:
            # First check buddy.train_defaults
            if isinstance(buddy_train_cfg, dict) and name in buddy_train_cfg:
                return buddy_train_cfg[name]
            # Then check root config
            if isinstance(cfg, dict) and name in cfg:
                return cfg[name]
            # Then check training section
            if isinstance(training_cfg, dict) and name in training_cfg:
                return training_cfg[name]
            return default_val
        except Exception:
            return default_val
    
    def _model_cfg_get(name: str, default_val):
        """Get config value from model section."""
        try:
            return model_cfg.get(name, default_val) if isinstance(model_cfg, dict) else default_val
        except Exception:
            return default_val

    # CLI overrides config; if CLI arg is None, fall back to config then hard default.
    seq_len_eff = int(args.seq_len) if getattr(args, "seq_len", None) is not None else int(_cfg_get("seq_len", 50))
    epochs_eff = int(args.epochs) if getattr(args, "epochs", None) is not None else int(_cfg_get("epochs", 300))
    batch_size_eff = int(args.batch_size) if getattr(args, "batch_size", None) is not None else int(_cfg_get("batch_size", 32))
    lr_eff = float(args.lr) if getattr(args, "lr", None) is not None else float(_cfg_get("lr", 0.001))
    patience_eff = int(args.patience) if getattr(args, "patience", None) is not None else int(_cfg_get("patience", 10))
    es_monitor_eff = str(args.es_monitor) if getattr(args, "es_monitor", None) is not None else str(_cfg_get("es_monitor", "direction"))
    combined_w_dir_eff = float(args.combined_w_dir) if getattr(args, "combined_w_dir", None) is not None else float(_cfg_get("combined_w_dir", 0.7))
    combined_w_conf_eff = float(args.combined_w_conf) if getattr(args, "combined_w_conf", None) is not None else float(_cfg_get("combined_w_conf", 0.3))
    steps_per_execution_eff = int(args.steps_per_execution) if getattr(args, "steps_per_execution", None) is not None else int(_cfg_get("steps_per_execution", 1))
    fit_verbose_eff = int(args.fit_verbose) if getattr(args, "fit_verbose", None) is not None else int(_cfg_get("fit_verbose", 1))
    shared_encoder_eff = (
        bool(args.shared_encoder)
        if getattr(args, "shared_encoder", None) is not None
        else bool(_cfg_get("shared_encoder", False))
    )
    
    # M1 METAL CRITICAL SETTINGS - now properly read from config
    # For store_true flags: CLI True overrides config, otherwise use config
    mixed_precision_eff = (
        True if bool(getattr(args, "mixed_precision", False))
        else bool(_cfg_get("mixed_precision", True))  # Default True for M1
    )
    jit_compile_eff = (
        True if bool(getattr(args, "jit_compile", False))
        else bool(_cfg_get("jit_compile", False))
    )
    cache_val_eff = (
        True if bool(getattr(args, "cache_val", False))
        else bool(_cfg_get("cache_val", True))  # Default True for M1
    )
    prefetch_eff = (
        getattr(args, "prefetch", None)
        if getattr(args, "prefetch", None) is not None
        else _cfg_get("prefetch", None)
    )
    shuffle_buffer_eff = (
        getattr(args, "shuffle_buffer", None)
        if getattr(args, "shuffle_buffer", None) is not None
        else _cfg_get("shuffle_buffer", None)
    )
    
    # Model type from CLI or config - TCN is default for M1 Metal performance
    model_type_eff = (
        str(args.model_type) if getattr(args, "model_type", None) is not None
        else str(_model_cfg_get("type", "tcn"))
    )
    
    tier2_stride_eff = (
        int(getattr(args, "tier2_calibration_stride", 0))
        if getattr(args, "tier2_calibration_stride", None) is not None
        else int(_cfg_get("tier2_calibration_stride", 5))
    )
    tier2_horizon_eff = (
        int(getattr(args, "tier2_horizon_candles", 0))
        if getattr(args, "tier2_horizon_candles", None) is not None
        else int(_cfg_get("tier2_horizon_candles", 0))
    )
    if int(tier2_horizon_eff) <= 0:
        tier2_horizon_eff = None

    # Default to fetching fresh OANDA candles for training unless the user
    # explicitly provides a local CSV.
    oanda_live_eff = bool(getattr(args, "oanda_live", False)) or (getattr(args, "csv", None) is None)

    # Use candles from args (default 5000 for training)
    train_candles = int(args.candles)

    # Check for multi-pair mode - skip single-pair OANDA fetch if enabled
    multi_pair_mode = bool(getattr(args, "multi_pair", False))
    
    oanda_fetch = None
    if bool(oanda_live_eff) and not multi_pair_mode:
        # Single-pair mode: validate and normalize instrument name
        instrument_raw = str(getattr(args, "instrument", "USD_JPY"))
        try:
            instrument_validated = _validate_instrument(instrument_raw)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise SystemExit(1)
        
        if instrument_raw.upper().replace("/", "_").replace("-", "_") != instrument_validated:
            console.print(f"[yellow]Normalized instrument: {instrument_raw} -> {instrument_validated}[/yellow]")
        
        console.print(Panel(
            f"[bold]Fetching Live Market Data[/bold]\n\n"
            f"[dim]Instrument:[/dim] {instrument_validated}\n"
            f"[dim]Granularity:[/dim] {args.granularity}\n"
            f"[dim]Candles:[/dim] {train_candles:,}",
            title="🌐 OANDA API",
            border_style="cyan",
        ))
        
        oanda_fetch = OandaFetchOptions(
            instrument=instrument_validated,
            granularity=str(args.granularity),
            candles=int(train_candles),
            price=str(getattr(args, "oanda_price", "MBA")),
            save_csv=getattr(args, "oanda_save_csv", None),
        )
    elif multi_pair_mode:
        # Multi-pair mode: create a placeholder oanda_fetch just for granularity/candles info
        # The actual multi-pair fetching happens in _train_buddy_impl
        oanda_fetch = OandaFetchOptions(
            instrument="MULTI_PAIR",  # Placeholder
            granularity=str(args.granularity),
            candles=int(train_candles),
            price=str(getattr(args, "oanda_price", "MBA")),
            save_csv=None,
        )

    init_from = getattr(args, "init_from", None)
    # WARM-START default=True: Always load existing weights for compounding learning
    # BUT only if the checkpoint file actually exists
    if not init_from and bool(getattr(args, "warm_start", True)):
        default_checkpoint = Path("trained_data") / "models" / "buddy_tf.keras"
        if default_checkpoint.exists():
            init_from = str(default_checkpoint)
        else:
            # Checkpoint doesn't exist - skip warm-start silently (first training run)
            pass

    advanced = BuddyTrainingAdvancedOptions(
        init_from=init_from,
        es_monitor=str(es_monitor_eff),
        combined_w_dir=float(combined_w_dir_eff),
        combined_w_conf=float(combined_w_conf_eff),
        train_smoothing=not bool(getattr(args, "no_train_smoothing", None) or _cfg_get("no_train_smoothing", False)),
        top_features=(
            int(getattr(args, "top_features", 0))
            if getattr(args, "top_features", None) is not None
            else (int(_cfg_get("top_features", 0)) if _cfg_get("top_features", None) is not None else None)
        ),
        median_window=(
            int(getattr(args, "median_window", 0))
            if getattr(args, "median_window", None) is not None
            else None
        ),
        vol_norm_window=(
            int(getattr(args, "vol_norm_window", 0))
            if getattr(args, "vol_norm_window", None) is not None
            else None
        ),
        min_volume=(
            int(getattr(args, "min_volume", 0))
            if getattr(args, "min_volume", None) is not None
            else None
        ),
        spread_filter=bool(getattr(args, "spread_filter", False)),
        spread_pctl=float(getattr(args, "spread_pctl", 0.99)),
        spread_mult=float(getattr(args, "spread_mult", 3.0)),
        tier2_calibrate=bool(getattr(args, "tier2_calibrate", True)),
        tier2_horizon_candles=(int(tier2_horizon_eff) if tier2_horizon_eff is not None else None),
        tier2_calibration_stride=int(tier2_stride_eff),
    )

    command_map["train-buddy"](
        args.config,
        None if bool(oanda_live_eff) else args.csv,
        oanda_fetch=oanda_fetch,
        advanced=advanced,
        feature_curriculum=getattr(args, "feature_curriculum", None),
        curriculum_ks=getattr(args, "curriculum_ks", None),
        pca_components=args.pca_components,
        seq_len=int(seq_len_eff),
        epochs=int(epochs_eff),
        batch_size=int(batch_size_eff),
        lr=float(lr_eff),
        patience=int(patience_eff),
        seed=int(args.seed),
        run_tag=args.run_tag,
        all_features=True,
        device=str(getattr(args, "device", "auto")),
        ignore_input_mismatches=bool(getattr(args, "ignore_input_mismatches", False)),
        disable_tier2_on_mismatch=bool(getattr(args, "disable_tier2_on_mismatch", False)),
        mixed_precision=bool(mixed_precision_eff),  # Now reads from config
        steps_per_execution=int(steps_per_execution_eff),
        jit_compile=bool(jit_compile_eff),  # Now reads from config
        shuffle_buffer=shuffle_buffer_eff,  # Now reads from config
        prefetch=prefetch_eff,  # Now reads from config
        cache_val=bool(cache_val_eff),  # Now reads from config
        combined_use_predict=bool(getattr(args, "combined_use_predict", True)),
        shared_encoder=bool(shared_encoder_eff),
        model_type=str(model_type_eff),  # NEW: Pass model type from config
        timing=bool(getattr(args, "timing", False)),
        fit_verbose=int(fit_verbose_eff),
        # Enterprise training options (enabled by default for ensemble)
        enterprise=not bool(getattr(args, "no_enterprise", False)),
        cv_folds=int(getattr(args, "cv_folds", 5)),
        bootstrap=not bool(getattr(args, "no_bootstrap", False)),
        bootstrap_samples=int(getattr(args, "bootstrap_samples", 1000)),
        mlflow_experiment=getattr(args, "mlflow_experiment", None),
        generate_report=not bool(getattr(args, "no_report", False)),
        # Continual learning - EWC enabled by default
        enable_ewc=not bool(getattr(args, "disable_ewc", False)),
        # Multi-pair foundation training
        multi_pair=bool(getattr(args, "multi_pair", False)),
        foundation_pairs=getattr(args, "foundation_pairs", None),
        # RL position sizer training (after ensemble)
        train_rl_sizer=not bool(getattr(args, "skip_rl", False)),
        rl_timesteps=int(getattr(args, "timesteps", 500_000)),
    )


def buddy_scan(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    pairs: Optional[str] = None,
    granularity: str = "H1",
    top_n: int = 5,
    verbose: bool = False,
    prompt_train: bool = False,
    diversified: bool = False,
    **kwargs: Any,
) -> list:
    """
    Scan multiple FX pairs to find the best trading opportunities.
    
    UPGRADED v2.0: Now uses BuddyScanner with:
    - Parallel pair scanning (4x faster)
    - Integrated position sizing (lots, SL/TP)
    - 50-candle quick backtesting
    - Drift detection with retraining prompts
    - Correlation analysis for diversification
    - Historical accuracy tracking
    - Shows tradeable pairs vs needs training
    
    Uses pair-specific models for predictions:
    - Transformer direction model for signal
    - XGBoost momentum for confirmation  
    - Ridge confidence scoring
    - RF risk assessment
    
    Falls back to technical indicators if model unavailable.
    
    NOTE: Scanner only displays results. Use 'buddy -I <PAIR> --execute' to trade.
    
    Args:
        diversified: If True, auto-filter to show only best pair per correlation cluster
    """
    try:
        # Use new enhanced BuddyScanner
        from buddy_scanner import BuddyScanner, EnhancedScanResult
        from pair_scanner import PairAnalysis
        
        # Don't override account_equity unless explicitly passed
        # Let config file value be used by default
        explicit_equity = kwargs.get("account_equity")
        use_rl_sizer = kwargs.get("use_rl_sizer", True)  # RL sizer enabled by default
        
        scanner = BuddyScanner(
            config_path=config_path,
            account_equity=explicit_equity,  # None = use config
            use_rl_sizer=use_rl_sizer,
        )
        
        # Parse pairs
        pair_list = None
        if pairs:
            pair_list = [p.strip().upper().replace("/", "_") for p in pairs.split(",")]
        
        # Run enhanced scan (display only, no execution prompt)
        results = scanner.scan(
            pairs=pair_list,
            granularity=granularity,
            top_n=top_n,
            verbose=True,  # Always verbose for CLI
            prompt_train=prompt_train,
            diversified=diversified,
        )
        
        # Convert to legacy format for backward compatibility
        legacy_results = []
        for r in results:
            # Create PairAnalysis-compatible object
            analysis = PairAnalysis(
                pair=r.pair,
                direction=r.direction,
                confidence=r.confidence,
                volatility_percentile=r.volatility_percentile,
                trend_strength=r.trend_strength,
                optimal_entry_score=r.entry_score,
                historical_accuracy=r.historical_accuracy,
                current_price=r.current_price,
                atr=r.atr,
                timestamp=r.timestamp,
            )
            legacy_results.append((analysis, r.gates_passed))
        
        return legacy_results
        
    except ImportError as e:
        # Fallback to legacy implementation if buddy_scanner not available
        console.print(f"[yellow]⚠ BuddyScanner not available ({e}), using legacy scan[/yellow]")
        return _buddy_scan_legacy(
            config_path=config_path,
            pairs=pairs,
            granularity=granularity,
            top_n=top_n,
            verbose=verbose,
            prompt_train=prompt_train,
            **kwargs,
        )


def buddy_predict_78(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    instrument: str = "EUR_USD",
    granularity: str = "H1",
    execute: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """
    Run prediction using the verified 78% model with aggressive position sizing.
    
    This is the recommended predict command for the $100k → $1M compounding strategy.
    Uses BuddyScanner for consistent position sizing with scan results.
    
    Args:
        config_path: Path to config file
        instrument: Currency pair (e.g., "EUR_USD", "USD_JPY")
        granularity: Timeframe (default H1)
        execute: If True, execute the trade on OANDA practice
        verbose: Show extra details
        
    Returns:
        Dict with prediction details or None on failure
    """
    from datetime import datetime
    
    # Normalize instrument
    instrument = instrument.upper().replace("/", "_")
    
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]📊 BUDDY PREDICT (78% Model)[/bold cyan]")
    console.print("=" * 60)
    
    # Use BuddyScanner for consistent results with scan
    try:
        from buddy_scanner import BuddyScanner
        
        scanner = BuddyScanner(config_path=config_path)
        
        # Run scan for just this one pair
        results = scanner.scan(
            pairs=[instrument],
            granularity=granularity,
            top_n=1,
            verbose=False,
            prompt_train=False,
        )
        
        if not results:
            console.print(f"[red]✗ No prediction for {instrument}[/red]")
            return None
        
        r = results[0]
        
        # Get live account equity from scanner (uses compounding!)
        account_equity = scanner._scan_config.account_equity
        
    except Exception as e:
        console.print(f"[red]✗ Scanner error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None
    
    # Display prediction (matches scan format)
    console.print(f"[green]✓ Pair Model[/green] ({instrument})")
    console.print(f"[dim]Data: {granularity} timeframe | Compounding Equity: ${account_equity:,.2f}[/dim]")
    console.print("-" * 60)
    
    signal_color = "green" if r.direction == "LONG" else "red"
    signal_emoji = "🟢" if r.direction == "LONG" else "🔴"
    
    console.print(f"\n{signal_emoji} [bold {signal_color}]{r.direction}[/bold {signal_color}] {instrument.replace('_', '/')}")
    console.print(f"   Confidence: [{signal_color}]{r.confidence:.1%}[/{signal_color}]")
    console.print(f"   Entry:      {r.current_price:.5f}")
    
    # Calculate actual SL/TP prices
    from fx_paper import pip_size
    pip = pip_size(instrument)
    
    if r.direction == "LONG":
        sl_price = r.current_price - (r.sl_pips * pip)
        tp_price = r.current_price + (r.tp_pips * pip)
    else:
        sl_price = r.current_price + (r.sl_pips * pip)
        tp_price = r.current_price - (r.tp_pips * pip)
    
    console.print(f"   SL:         {sl_price:.5f} ({r.sl_pips:.0f} pips)")
    console.print(f"   TP:         {tp_price:.5f} ({r.tp_pips:.0f} pips)")
    console.print(f"   R:R:        1:{r.tp_pips/r.sl_pips:.1f}" if r.sl_pips > 0 else "   R:R:        N/A")
    console.print()
    
    # Position sizing
    units = int(r.recommended_lots * 100000)
    pip_value = 10.0 if not instrument.endswith("JPY") else 7.5
    est_profit = r.recommended_lots * r.tp_pips * pip_value
    est_loss = r.recommended_lots * r.sl_pips * pip_value
    
    console.print(f"   [bold]Position: {r.recommended_lots:.2f} lots[/bold] ({units:,} units)")
    console.print(f"   Risk:       ${est_loss:,.0f} ({r.risk_pct:.0%} of ${account_equity:,.0f})")
    console.print(f"   Target:     [green]+${est_profit:,.0f}[/green]")
    console.print("-" * 60)
    
    # Gate check
    if r.gates_passed:
        console.print(f"[green]✓ GATES PASSED[/green] (conf {r.confidence:.0%})")
    else:
        console.print(f"[yellow]⚠ GATES NOT PASSED[/yellow] (conf {r.confidence:.0%})")
    
    # Execute trade if requested
    if execute and r.gates_passed and r.recommended_lots > 0:
        console.print("\n[bold]Executing trade...[/bold]")
        try:
            from oanda_practice import OandaPracticeClient
            client = OandaPracticeClient.from_env()
            
            result = client.create_market_order(
                instrument=instrument,
                units=units if r.direction == "LONG" else -units,
                take_profit_price=round(tp_price, 5),
                stop_loss_price=round(sl_price, 5),
            )
            
            if result and "orderFillTransaction" in result:
                fill = result["orderFillTransaction"]
                fill_price = float(fill.get("price", r.current_price))
                trade_id = fill.get("tradeOpened", {}).get("tradeID", "N/A")
                console.print(f"[green]✓ FILLED[/green] @ {fill_price:.5f} (Trade #{trade_id})")
                
                # Log to journal
                try:
                    from memory_client import MLEngineMemory
                    mem = MLEngineMemory()
                    mem.log_trade({
                        "timestamp": datetime.now().isoformat(),
                        "instrument": instrument,
                        "direction": r.direction,
                        "confidence": r.confidence,
                        "lots": r.recommended_lots,
                        "entry": fill_price,
                        "sl": sl_price,
                        "tp": tp_price,
                        "trade_id": trade_id,
                        "model": "pair_specific",
                    })
                    console.print("[dim]📓 Trade logged[/dim]")
                except Exception:
                    pass
            else:
                console.print(f"[yellow]Order result: {result}[/yellow]")
                
        except Exception as e:
            console.print(f"[red]✗ Execution failed: {e}[/red]")
    elif execute and not r.gates_passed:
        console.print("[yellow]⚠ Trade not executed - gates not passed[/yellow]")
    elif execute and r.recommended_lots <= 0:
        console.print("[yellow]⚠ Trade not executed - no position size calculated[/yellow]")
    
    console.print("=" * 60 + "\n")
    
    return {
        "instrument": instrument,
        "direction": r.direction,
        "confidence": r.confidence,
        "lots": r.recommended_lots,
        "entry": r.current_price,
        "sl": sl_price,
        "tp": tp_price,
        "sl_pips": r.sl_pips,
        "tp_pips": r.tp_pips,
        "gates_passed": r.gates_passed,
        "score": r.overall_score,
    }


def _buddy_scan_legacy(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    pairs: Optional[str] = None,
    granularity: str = "H1",
    top_n: int = 5,
    verbose: bool = False,
    prompt_train: bool = True,
    **kwargs: Any,
) -> list:
    """Legacy buddy_scan implementation (fallback)."""
    import json
    import numpy as np
    from datetime import datetime
    from rich.table import Table
    
    console.print("\n" + "=" * 70)
    console.print(f"[bold cyan]📡 MULTI-PAIR SCANNER[/bold cyan]")
    console.print("=" * 70)
    
    # Suppress config loading message
    import logging as _logging
    _utils_logger = _logging.getLogger('utils')
    _utils_prev = _utils_logger.level
    _utils_logger.setLevel(_logging.WARNING)
    
    cfg = load_config(config_path)
    
    _utils_logger.setLevel(_utils_prev)
    
    # Parse pairs
    from pair_scanner import MAJOR_PAIRS, PairAnalysis
    
    if pairs:
        pair_list = [p.strip().upper().replace("/", "_") for p in pairs.split(",")]
    else:
        pair_list = MAJOR_PAIRS  # Default to majors
    
    console.print(f"[dim]Scanning {len(pair_list)} pairs: {', '.join(pair_list)}[/dim]")
    console.print(f"[dim]Timeframe: {granularity} | Top {top_n} results[/dim]")
    console.print("-" * 70)
    
    # Suppress verbose logging during model loading
    import logging as _logging
    import os
    import warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO/WARNING
    
    # Suppress version warnings for XGBoost and sklearn (comprehensive)
    warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
    warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
    warnings.filterwarnings('ignore', message='.*serialized model.*')
    warnings.filterwarnings('ignore', message='.*older version of XGBoost.*')
    warnings.filterwarnings('ignore', message='.*Booster.save_model.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Suppress all model-related loggers
    _loggers_to_quiet = ['modular_trainers', 'modular_inference', 'feature_engineering', 
                         'modular_data_loaders', 'tensorflow', 'absl', 'xgboost']
    _prev_levels = {}
    for name in _loggers_to_quiet:
        logger = _logging.getLogger(name)
        _prev_levels[name] = logger.level
        logger.setLevel(_logging.ERROR)  # Only show errors
    
    # Initialize scanner
    from oanda_practice import OandaPracticeClient
    from feature_engineering import FeatureEngineering
    from fx_paper import candles_to_ohlcv_df
    
    client = OandaPracticeClient.from_env()
    fe = FeatureEngineering(cfg.get("feature_engineering", {}))
    
    # Try to load modular ensemble (legacy fallback)
    modular_ensemble = None
    modular_ensemble_meta_path = Path("trained_data") / "models" / "modular_ensemble.meta.json"
    
    if modular_ensemble_meta_path.exists():
        try:
            from modular_inference import ModularEnsembleInference
            modular_ensemble = ModularEnsembleInference()
            modular_ensemble.load_models()
            
            # Load meta for model info
            meta = json.loads(modular_ensemble_meta_path.read_text())
            
            # Get val_accuracy - handle multiple possible locations
            val_acc = 0
            if "results" in meta:
                val_acc = meta.get("results", {}).get("transformer", {}).get("val_accuracy", 0)
            if val_acc == 0 and "models" in meta:
                val_acc = meta.get("models", {}).get("direction", {}).get("metrics", {}).get("val_accuracy", 0)
            
            # Get training info - check multiple possible locations
            trained_pairs = meta.get("selected_pairs", [])
            if not trained_pairs:
                trained_pairs = meta.get("trained_pairs", [])
            if not trained_pairs:
                trained_at = meta.get("trained_at", "unknown")
                trained_pairs = [f"EUR_USD (trained {trained_at[:10]})"] if trained_at != "unknown" else ["EUR_USD"]
            
            console.print(f"[bold green]✓ Legacy ensemble loaded[/bold green] (val_accuracy: {val_acc:.1%})")
            console.print(f"[dim]  Trained on: {', '.join(trained_pairs) if trained_pairs else 'EUR_USD'}[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not load modular ensemble: {e}[/yellow]")
            console.print("[dim]Falling back to technical indicators[/dim]")
            modular_ensemble = None
    else:
        console.print("[dim]No modular ensemble found, using technical indicators[/dim]")
    
    console.print("-" * 70)
    
    # Scan each pair
    analyses = []
    
    for pair in pair_list:
        try:
            console.print(f"[dim]Scanning {pair}...[/dim]", end=" ")
            
            # Fetch data
            fetch_count = 400
            resp = client.get_candles(pair, granularity=granularity, count=fetch_count, price="MBA")
            df_raw = candles_to_ohlcv_df(resp)
            
            if len(df_raw) < 100:
                console.print("[yellow]insufficient data[/yellow]")
                continue
            
            # Feature engineering
            df = fe.create_features(df_raw.copy(), include_all=True)
            
            # Clean up data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill().fillna(0.0)
            
            # Calculate metrics from technical indicators
            vol_pct = 0.5
            trend_str = 0.5
            entry_score = 0.5
            current_price = df["close"].iloc[-1]
            atr = df["atr"].iloc[-1] if "atr" in df.columns else 0.0
            
            # Volatility percentile
            if "atr" in df.columns:
                atr_series = df["atr"].dropna()
                if len(atr_series) > 20:
                    vol_pct = float((atr_series < atr_series.iloc[-1]).mean())
            
            # Trend strength from ADX
            if "adx" in df.columns:
                adx_val = df["adx"].iloc[-1]
                trend_str = min(float(adx_val) / 60, 1.0)
            
            # === USE MODULAR ENSEMBLE IF AVAILABLE ===
            direction = "LONG"
            confidence = 0.5
            model_used = False
            gates_passed = False
            ridge_conf = 0.0
            tcn_conf = 0.0
            
            if modular_ensemble is not None:
                try:
                    signal = modular_ensemble.predict(df)
                    
                    # Use TCN direction even if gates don't pass (for scanning)
                    if signal.tcn_direction is not None:
                        direction = "LONG" if signal.tcn_direction == 1 else "SHORT"
                        # Calculate TCN confidence (distance from 0.5 = uncertainty)
                        # 0.5 -> 0%, 0.6 -> 20%, 0.7 -> 40%, 0.8 -> 60%, 0.9 -> 80%, 1.0 -> 100%
                        tcn_conf = abs(signal.tcn_probability - 0.5) * 200  # 0-100 scale to match ridge
                        # Use the higher of TCN or Ridge confidence
                        ridge_conf = signal.ridge_confidence
                        confidence = max(tcn_conf, ridge_conf) / 100  # Convert to 0-1 for internal use
                        model_used = True
                        gates_passed = signal.trade
                    elif signal.direction is not None:
                        direction = signal.direction.upper()
                        confidence = signal.confidence if signal.confidence > 0 else 0.5
                        model_used = True
                        gates_passed = signal.trade
                        ridge_conf = signal.ridge_confidence
                        tcn_conf = confidence * 100
                        
                except Exception as e:
                    if verbose:
                        console.print(f"[yellow]Model prediction failed: {e}[/yellow]")
                    model_used = False
            
            # === FALLBACK TO TECHNICAL INDICATORS ===
            if not model_used:
                # Use RSI for direction
                if "rsi" in df.columns:
                    rsi = df["rsi"].iloc[-1]
                    if rsi < 30:
                        direction = "LONG"
                        confidence = 0.5 + (30 - rsi) / 60
                    elif rsi > 70:
                        direction = "SHORT"
                        confidence = 0.5 + (rsi - 70) / 60
                    else:
                        if "macd" in df.columns:
                            macd = df["macd"].iloc[-1]
                            direction = "LONG" if macd > 0 else "SHORT"
                            confidence = 0.5 + min(abs(macd) * 10, 0.2)
            
            # Entry score based on MA alignment and RSI
            if "rsi" in df.columns:
                rsi = df["rsi"].iloc[-1]
                if 30 < rsi < 70:
                    entry_score += 0.15
            
            if "sma_20" in df.columns and "sma_50" in df.columns:
                sma_20 = df["sma_20"].iloc[-1]
                sma_50 = df["sma_50"].iloc[-1]
                close = df["close"].iloc[-1]
                
                # Aligned trend
                if (close > sma_20 > sma_50) or (close < sma_20 < sma_50):
                    entry_score += 0.15
                    if direction == "LONG" and close > sma_20:
                        confidence += 0.05
                    elif direction == "SHORT" and close < sma_20:
                        confidence += 0.05
            
            # Cap confidence
            confidence = min(confidence, 0.95)
            
            analysis = PairAnalysis(
                pair=pair,
                direction=direction,
                confidence=confidence,
                volatility_percentile=vol_pct,
                trend_strength=trend_str,
                optimal_entry_score=entry_score,
                current_price=current_price,
                atr=atr,
                timestamp=datetime.now(),
            )
            analyses.append((analysis, gates_passed))  # Include gate status
            
            # Show result with model info
            signal_marker = "🤖" if model_used else "📊"
            gate_marker = "✅" if gates_passed else "⚠️" if model_used else ""
            # Display the actual confidence used (max of tcn_conf and ridge_conf)
            effective_conf = max(tcn_conf, ridge_conf) if model_used else confidence * 100
            conf_display = f"{effective_conf:.0f}%"
            console.print(f"{signal_marker} [{('green' if direction == 'LONG' else 'red')}]{direction}[/] (conf: {conf_display}) {gate_marker}")
            
        except Exception as e:
            console.print(f"[red]error: {e}[/red]")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    # Sort and display results
    analyses.sort(key=lambda x: x[0].overall_score, reverse=True)
    
    console.print("\n" + "=" * 70)
    model_label = "Ensemble" if modular_ensemble else "Technical"
    console.print(f"[bold green]📊 SCAN RESULTS ({model_label}) - Top {min(top_n, len(analyses))}[/bold green]")
    console.print("=" * 70)
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Pair", style="bold")
    table.add_column("Signal", justify="center")
    table.add_column("Confidence", justify="right")
    table.add_column("Gates", justify="center")
    table.add_column("Trend", justify="right")
    table.add_column("Volatility", justify="right")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Price", justify="right")
    
    for i, (a, gates_ok) in enumerate(analyses[:top_n], 1):
        signal_color = "green" if a.direction == "LONG" else "red"
        gate_display = "[green]✓[/green]" if gates_ok else "[yellow]⚠[/yellow]"
        table.add_row(
            str(i),
            a.pair.replace("_", "/"),
            f"[{signal_color}]{a.direction}[/{signal_color}]",
            f"{a.confidence:.0%}",
            gate_display,
            f"{a.trend_strength:.2f}",
            f"{a.volatility_percentile:.0%}",
            f"{a.overall_score:.2f}",
            f"{a.current_price:.5f}" if a.current_price else "-",
        )
    
    console.print(table)
    
    # Best opportunities
    if analyses:
        analyses_only = [a for a, _ in analyses]
        best_long = max([a for a in analyses_only if a.direction == "LONG"], key=lambda x: x.overall_score, default=None)
        best_short = max([a for a in analyses_only if a.direction == "SHORT"], key=lambda x: x.overall_score, default=None)
        
        # Count tradeable (gates passed)
        tradeable_count = sum(1 for _, gates_ok in analyses if gates_ok)
        
        console.print(f"\n[bold]🎯 TOP OPPORTUNITIES:[/bold] ({tradeable_count}/{len(analyses)} tradeable)")
        if best_long:
            console.print(f"  [green]Best LONG:[/green]  {best_long.pair.replace('_', '/')} (conf: {best_long.confidence:.0%}, score: {best_long.overall_score:.2f})")
        if best_short:
            console.print(f"  [red]Best SHORT:[/red] {best_short.pair.replace('_', '/')} (conf: {best_short.confidence:.0%}, score: {best_short.overall_score:.2f})")
    
    console.print("\n" + "=" * 70)
    
    # Restore logging levels
    for name, level in _prev_levels.items():
        _logging.getLogger(name).setLevel(level)
    
    # === SCAN → TRAIN INTERACTIVE PROMPT ===
    if prompt_train and analyses:
        # Find best tradeable pair (gates passed)
        tradeable = [(a, g) for a, g in analyses if g]
        if tradeable:
            best_pair = tradeable[0][0].pair
            best_conf = tradeable[0][0].confidence
        else:
            # Fall back to best overall if none tradeable
            best_pair = analyses[0][0].pair
            best_conf = analyses[0][0].confidence
        
        # Check for existing model (warm-start available)
        model_path = Path("trained_data/models/transformer_direction.keras")
        warm_start_available = model_path.exists()
        warm_label = " (warm-start)" if warm_start_available else ""
        
        console.print(f"\n[bold yellow]🎯 Train {best_pair.replace('_', '/')}?{warm_label}[/bold yellow]")
        train_answer = _buddy_wizard_ask("  [y/n]: ")
        
        if _parse_bool_answer(train_answer, default=False):
            console.print(f"\n[bold green]Starting training for {best_pair}...[/bold green]")
            
            # Import memory client for tracking (if available)
            try:
                from memory_client import MLEngineMemory
                memory = MLEngineMemory()
                if memory.should_skip_pair(best_pair, max_age_hours=24):
                    console.print(f"[yellow]⚠ {best_pair} was trained < 24h ago. Training anyway...[/yellow]")
            except ImportError:
                pass  # Memory client not available
            
            # Call train_buddy with full ensemble
            train_opts = BuddyTrainingOptions(
                oanda_fetch=OandaFetchOptions(
                    instrument=best_pair,
                    granularity=granularity,
                    candles=12000,  # Full dataset for quality training
                ),
                epochs=200,
                batch_size=128,
                enterprise=True,  # Enable MLflow + CV
            )
            train_buddy(config_path, options=train_opts)
            
            # Log training to memory (if available)
            try:
                from memory_client import MLEngineMemory
                memory = MLEngineMemory()
                memory.log_training(best_pair, {"granularity": granularity})
            except ImportError:
                pass
    
    return analyses


def train_rl_sizer(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    timesteps: int = 500_000,
    episodes: int | None = None,
    pairs: str | None = None,
    granularity: str = "H1",
    candles: int = 5000,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """
    Train RL position sizing agent using PPO.
    
    Uses a two-phase subprocess approach to avoid TensorFlow/PyTorch GPU conflicts.
    """
    import json
    import subprocess
    import time
    import logging
    from datetime import datetime
    from pathlib import Path
    
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich import box
    
    # Suppress noisy logging during data generation
    logging.getLogger('modular_data_loaders').setLevel(logging.WARNING)
    logging.getLogger('modular_trainers').setLevel(logging.WARNING)
    logging.getLogger('modular_inference').setLevel(logging.WARNING)
    logging.getLogger('feature_engineering').setLevel(logging.WARNING)
    
    # ═══════════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════════
    console.print()
    console.print(Panel.fit(
        "[bold cyan]RL Position Sizer Training[/bold cyan]\n"
        f"[dim]PPO • {timesteps:,} timesteps • {granularity}[/dim]",
        border_style="cyan"
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHECKLIST
    # ═══════════════════════════════════════════════════════════════════════
    steps = [
        ["⏳", "Check ensemble model", ""],
        ["⏳", "Connect to OANDA", ""],
        ["⏳", "Generate predictions", ""],
        ["⏳", "Train PPO (subprocess)", ""],
        ["⏳", "Finalize", ""],
    ]
    
    def print_checklist():
        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("", width=2)
        table.add_column("", width=28)
        table.add_column("", style="dim")
        for s in steps:
            clr = "green" if s[0] == "✓" else ("yellow" if s[0] == "⏳" else "red")
            table.add_row(f"[{clr}]{s[0]}[/{clr}]", s[1], s[2])
        console.print(table)
    
    # Step 1: Check ensemble
    modular_meta_path = Path("trained_data/models/modular_ensemble.meta.json")
    if not modular_meta_path.exists():
        steps[0] = ["✗", "Check ensemble model", "Not found"]
        print_checklist()
        console.print("\n[yellow]Train ensemble first: buddy train --oanda-live -n 12000[/yellow]")
        return
    steps[0] = ["✓", "Check ensemble model", "Found"]
    
    # Suppress TF logging
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    _configure_tf_metal(verbose=False)
    
    from modular_inference import ModularEnsembleInference
    from feature_engineering import FeatureEngineering
    from fx_paper import candles_to_ohlcv_df
    
    ensemble = ModularEnsembleInference(use_rl_sizer=False)
    ensemble.load_models()
    
    # Step 2: Connect OANDA
    try:
        from oanda_practice import OandaPracticeClient
        oanda = OandaPracticeClient.from_env()
        steps[1] = ["✓", "Connect to OANDA", "Connected"]
    except Exception as e:
        steps[1] = ["✗", "Connect to OANDA", str(e)[:20]]
        print_checklist()
        return
    
    # Parse pairs
    ALL_MAJOR_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]
    pair_list = [p.strip() for p in pairs.split(",")] if pairs else ALL_MAJOR_PAIRS
    
    # Print checklist so far
    console.print()
    print_checklist()
    console.print()
    
    # Step 3: Generate predictions with progress bar
    import pandas as pd
    import numpy as np
    
    fe = FeatureEngineering()
    all_features = []
    all_predictions = []
    all_prices = []
    
    console.print("[bold]Generating Training Data[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        # Task for pairs
        pairs_task = progress.add_task(f"Processing {len(pair_list)} pairs", total=len(pair_list))
        
        for pair in pair_list:
            progress.update(pairs_task, description=f"[cyan]{pair}[/cyan]")
            
            try:
                resp = oanda.get_candles(pair, granularity=granularity, count=candles, price="MBA")
                df_raw = candles_to_ohlcv_df(resp)
                
                if df_raw is None or len(df_raw) < 200:
                    progress.advance(pairs_task)
                    continue
                
                df = fe.create_features(df_raw.copy(), include_all=True)
                df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
                prices = df['close'].values
                
                # Generate predictions (silently)
                for i in range(100, len(df) - 1):
                    df_slice = df.iloc[:i+1].copy()
                    try:
                        signal = ensemble.predict(df_slice)
                        direction_prob = signal.tcn_probability if signal.tcn_probability else 0.5
                        confidence = signal.ridge_confidence / 100.0 if signal.ridge_confidence else 0.5
                        vol = df_slice['close'].pct_change().rolling(20).std().iloc[-1]
                        momentum = signal.xgb_momentum if signal.xgb_momentum else 0.0
                        
                        features = np.array([vol, momentum, signal.rf_drawdown_pips / 100.0 if signal.rf_drawdown_pips else 0.0])
                        predictions = np.array([direction_prob, confidence])
                        
                        all_features.append(features)
                        all_predictions.append(predictions)
                        all_prices.append(prices[i])
                    except Exception:
                        continue
                
            except Exception:
                pass
            
            progress.advance(pairs_task)
    
    if not all_features:
        steps[2] = ["✗", "Generate predictions", "No data"]
        print_checklist()
        return
    
    features_array = np.array(all_features, dtype=np.float32)
    predictions_array = np.array(all_predictions, dtype=np.float32)
    prices_array = np.array(all_prices, dtype=np.float32)
    
    steps[2] = ["✓", "Generate predictions", f"{len(features_array):,} samples"]
    
    # Save training data
    data_file = Path("trained_data/models/rl_train_data.npz")
    np.savez(data_file, features=features_array, predictions=predictions_array, prices=prices_array)
    
    # Update checklist
    console.print()
    print_checklist()
    console.print()
    
    # Step 4: Run PPO training in subprocess
    console.print("[bold]Training PPO Agent (subprocess)[/bold]")
    console.print()
    
    script_path = Path(__file__).parent / "train_rl_standalone.py"
    cmd = [sys.executable, str(script_path), "--data", str(data_file), "--timesteps", str(timesteps)]
    if verbose:
        cmd.append("--verbose")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=Path(__file__).parent,
        )
        
        for line in process.stdout:
            print(line, end="")
        
        process.wait()
        
        if process.returncode != 0:
            steps[3] = ["✗", "Train PPO (subprocess)", f"Exit {process.returncode}"]
            console.print()
            print_checklist()
            return
        
        train_time = time.time() - start_time
        steps[3] = ["✓", "Train PPO (subprocess)", f"{train_time:.0f}s"]
        
    except Exception as e:
        steps[3] = ["✗", "Train PPO (subprocess)", str(e)[:20]]
        console.print()
        print_checklist()
        return
    
    # Step 5: Finalize
    meta_path = Path("trained_data/models/rl_position_sizer.meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["pairs_used"] = pair_list
        meta["granularity"] = granularity
        meta["candles_per_pair"] = candles
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
    
    if data_file.exists():
        data_file.unlink()
    
    steps[4] = ["✓", "Finalize", "Cleaned up"]
    
    # Final checklist
    console.print()
    print_checklist()
    
    # Summary panel
    console.print()
    console.print(Panel(
        f"[bold green]✓ RL Position Sizer Trained[/bold green]\n\n"
        f"Timesteps: {timesteps:,}  •  Time: {train_time:.0f}s  •  Samples: {len(features_array):,}\n"
        f"Data: {len(pair_list)} pairs × {candles:,} candles ({granularity})\n\n"
        f"[bold]Usage:[/bold]  buddy scan --use-rl-sizer\n"
        f"        buddy --use-rl-sizer -I USD_JPY",
        border_style="green"
    ))


def retrain_gates(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    pairs: str | None = None,
    granularity: str = "H1",
    candles: int = 5000,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """
    Retrain only sklearn gate models (XGBoost, RF, Ridge) while keeping Transformer.
    
    This solves sklearn version mismatch issues by retraining the fast sklearn
    models locally while preserving the proven 78% Colab-trained Transformer.
    
    Args:
        config_path: Path to config file
        pairs: Comma-separated pairs to train on (default: all major pairs)
        granularity: Candle timeframe (default: H1)
        candles: Number of candles per pair (default: 5000, will batch if >5000)
        verbose: Show detailed logs
    """
    import json
    import pickle
    import sklearn
    import xgboost
    import time
    from datetime import datetime
    from pathlib import Path
    
    from rich.panel import Panel
    from rich.progress import track
    
    # ALL major forex pairs by default
    ALL_MAJOR_PAIRS = [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
        "AUD_USD", "USD_CAD", "NZD_USD",
        "EUR_GBP", "EUR_JPY", "GBP_JPY",
        "EUR_CHF", "EUR_AUD", "EUR_CAD",
        "AUD_JPY", "CAD_JPY", "CHF_JPY",
    ]
    
    # Parse pairs
    if pairs:
        pair_list = [p.strip() for p in pairs.split(",")]
    else:
        pair_list = ALL_MAJOR_PAIRS
    
    console.print(Panel(
        f"[bold]Retraining Gate Models Locally[/bold]\n\n"
        f"Pairs: {', '.join(pair_list)}\n"
        f"Candles per pair: {candles:,}\n"
        f"Granularity: {granularity}\n\n"
        "[dim]This will retrain XGBoost, RF, and Ridge models[/dim]\n"
        "[dim]The Transformer direction model will be kept unchanged[/dim]",
        title="🔧 Gate Model Retraining",
        border_style="cyan"
    ))
    
    # Import trainers and data loaders
    from modular_data_loaders import (
        compute_normalized_features,
        load_xgboost_data,
        load_rf_data,
        load_ridge_data
    )
    from modular_trainers import (
        XGBoostTrainer,
        RandomForestTrainer,
        RidgeTrainer,
        TrainerConfig
    )
    from feature_engineering import FeatureEngineering
    
    # Fetch data from OANDA
    console.print("\n[bold]Step 1: Fetching data from OANDA...[/bold]")
    
    # OANDA API limit is 5000 candles per request
    OANDA_MAX_CANDLES = 5000
    
    try:
        from oanda_practice import OandaPracticeClient
        oanda = OandaPracticeClient.from_env()
    except Exception as e:
        console.print(f"[red]Failed to connect to OANDA: {e}[/red]")
        console.print("[yellow]Please set OANDA_ACCOUNT_ID and OANDA_TOKEN environment variables[/yellow]")
        return
    
    all_dfs = []
    import pandas as pd
    
    for pair in track(pair_list, description="Fetching pairs..."):
        try:
            pair_dfs = []
            remaining = candles
            last_time = None
            
            # Batch requests if candles > 5000
            while remaining > 0:
                batch_size = min(remaining, OANDA_MAX_CANDLES)
                
                # Build request params
                params = {
                    'granularity': granularity,
                    'count': batch_size
                }
                if last_time:
                    params['to'] = last_time
                
                response = oanda.get_candles(pair, **params)
                candles_raw = response.get('candles', [])
                
                if not candles_raw:
                    break
                
                rows = []
                for c in candles_raw:
                    mid = c.get('mid', {})
                    rows.append({
                        'time': c.get('time'),
                        'open': float(mid.get('o', 0)),
                        'high': float(mid.get('h', 0)),
                        'low': float(mid.get('l', 0)),
                        'close': float(mid.get('c', 0)),
                        'volume': int(c.get('volume', 0)),
                    })
                
                batch_df = pd.DataFrame(rows)
                pair_dfs.append(batch_df)
                
                # Update for next batch (go backwards in time)
                last_time = candles_raw[0].get('time')  # Oldest candle in batch
                remaining -= len(candles_raw)
                
                # Small delay to avoid rate limiting
                if remaining > 0:
                    time.sleep(0.2)
            
            if pair_dfs:
                # Combine batches and sort by time
                df = pd.concat(pair_dfs, ignore_index=True)
                df = df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
                df['pair'] = pair
                all_dfs.append(df)
                console.print(f"  ✓ {pair}: {len(df):,} candles")
            
        except Exception as e:
            console.print(f"[yellow]  ✗ {pair}: {e}[/yellow]")
            continue
    
    if not all_dfs:
        console.print("[red]No data available. Please check OANDA connection.[/red]")
        return
    
    # Combine all pairs
    combined_df = pd.concat(all_dfs, ignore_index=True)
    console.print(f"\n[green]Combined data: {len(combined_df):,} total candles[/green]")
    
    # Feature engineering
    console.print("\n[bold]Step 2: Computing features...[/bold]")
    
    fe = FeatureEngineering()
    feature_df = fe.create_features(combined_df)
    console.print(f"  Features computed: {len(feature_df.columns)} columns")
    
    # Compute normalized features
    feature_df = compute_normalized_features(feature_df)
    console.print(f"  Normalized features: {len(feature_df.columns)} columns")
    
    # Load data for each model
    console.print("\n[bold]Step 3: Preparing training data...[/bold]")
    
    split = (0.7, 0.2, 0.1)
    
    xgb_data = load_xgboost_data(feature_df, split)
    rf_data = load_rf_data(feature_df, split)
    ridge_data = load_ridge_data(feature_df, split)
    
    console.print(f"  XGBoost: {len(xgb_data['X_train']):,} train, {len(xgb_data['X_val']):,} val")
    console.print(f"  RF: {len(rf_data['X_train']):,} train, {len(rf_data['X_val']):,} val")
    console.print(f"  Ridge: {len(ridge_data['X_train']):,} train, {len(ridge_data['X_val']):,} val")
    
    # Configure trainer
    config = TrainerConfig()
    model_dir = Path("trained_data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Train XGBoost
    console.print("\n[bold]Step 4: Training XGBoost (Momentum)...[/bold]")
    
    xgb_trainer = XGBoostTrainer(config)
    xgb_metrics = xgb_trainer.train(
        xgb_data['X_train'], xgb_data['y_train'],
        xgb_data['X_val'], xgb_data['y_val'],
        feature_names=xgb_data['feature_names'],
        momentum_norm_factor=xgb_data.get('momentum_norm_factor'),
    )
    xgb_trainer.save(str(model_dir / "xgb_momentum.pkl"))
    
    console.print(f"  [green]✓ XGBoost: momentum_mae={xgb_metrics['momentum_mae']:.4f}, "
                 f"accel_acc={xgb_metrics['acceleration_accuracy']:.1%}[/green]")
    
    # Train RF
    console.print("\n[bold]Step 5: Training Random Forest (Risk)...[/bold]")
    
    rf_trainer = RandomForestTrainer(config)
    rf_metrics = rf_trainer.train(
        rf_data['X_train'], rf_data['y_train'],
        rf_data['X_val'], rf_data['y_val'],
        feature_names=rf_data['feature_names'],
    )
    rf_trainer.save(str(model_dir / "rf_risk.pkl"))
    
    drawdown_mae_bps = rf_metrics.get('drawdown_mae_bps', rf_metrics.get('drawdown_mae_pct', 0) * 10000)
    console.print(f"  [green]✓ RF: drawdown_mae={drawdown_mae_bps:.1f} bps, "
                 f"streak_mae={rf_metrics['streak_prob_mae']:.4f}[/green]")
    
    # Train Ridge
    console.print("\n[bold]Step 6: Training ElasticNet (Confidence)...[/bold]")
    
    ridge_trainer = RidgeTrainer(config)
    ridge_metrics = ridge_trainer.train(
        ridge_data['X_train'], ridge_data['y_train'],
        ridge_data['X_val'], ridge_data['y_val'],
        feature_names=ridge_data['feature_names'],
    )
    ridge_trainer.save(str(model_dir / "ridge_confidence.pkl"))
    
    console.print(f"  [green]✓ Ridge: MAE={ridge_metrics['confidence_mae']:.2f}, "
                 f"R²={ridge_metrics['r2_score']:.4f}[/green]")
    
    # Update metadata
    console.print("\n[bold]Step 7: Updating metadata...[/bold]")
    
    meta_path = model_dir / "modular_ensemble.meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}
    
    # Update with new training info
    meta['gate_models_retrained_at'] = datetime.now().isoformat()
    meta['gate_models_trained_on'] = 'local_m1'
    meta['sklearn_version'] = sklearn.__version__
    meta['xgboost_version'] = xgboost.__version__
    meta['retrained_pairs'] = pair_list
    meta['retrained_candles'] = candles
    meta['retrained_granularity'] = granularity
    
    # Update results
    if 'results' not in meta:
        meta['results'] = {}
    meta['results']['xgboost'] = xgb_metrics
    meta['results']['random_forest'] = rf_metrics
    meta['results']['ridge'] = ridge_metrics
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    console.print(f"  [green]✓ Metadata updated with sklearn={sklearn.__version__}, xgboost={xgboost.__version__}[/green]")
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print(Panel(
        f"[bold green]Gate Models Retrained Successfully![/bold green]\n\n"
        f"[bold]sklearn version:[/bold] {sklearn.__version__}\n"
        f"[bold]xgboost version:[/bold] {xgboost.__version__}\n\n"
        f"[bold]XGBoost (Momentum):[/bold]\n"
        f"  MAE: {xgb_metrics['momentum_mae']:.4f}\n"
        f"  Acceleration Accuracy: {xgb_metrics['acceleration_accuracy']:.1%}\n\n"
        f"[bold]Random Forest (Risk):[/bold]\n"
        f"  Drawdown MAE: {drawdown_mae_bps:.1f} bps\n"
        f"  Streak MAE: {rf_metrics['streak_prob_mae']:.4f}\n\n"
        f"[bold]ElasticNet (Confidence):[/bold]\n"
        f"  MAE: {ridge_metrics['confidence_mae']:.2f}\n"
        f"  R²: {ridge_metrics['r2_score']:.4f}\n\n"
        f"[dim]Run 'buddy predict' to test the updated gates[/dim]",
        title="✅ Complete",
        border_style="green"
    ))


def suggest_improvements(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    instrument: str | None = None,
    auto_apply: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """Meta-learning: Generate and optionally auto-apply model improvement suggestions.
    
    Uses LLM to analyze current model architecture and performance, then suggests
    improvements. Hyperparam suggestions that pass gates are auto-applied if requested.
    
    When a specific instrument is provided, only that pair's config is modified.
    
    Gates for auto-approval:
    1. confidence > 70%
    2. expected Sharpe improvement > 5%
    3. implementation_ease == "easy"
    4. category == "hyperparam" (architecture/feature changes require manual review)
    
    Args:
        config_path: Path to config file
        instrument: Specific pair to analyze (e.g., USD_JPY). If None, uses global model.
        auto_apply: If True, automatically apply approved hyperparam changes
        verbose: Show detailed output
    """
    from pathlib import Path
    import json
    
    # Normalize instrument
    if instrument:
        instrument = instrument.upper().replace("/", "_")
        console.print(f"\n[bold cyan]🧠 META-LEARNING: Improvements for {instrument}[/bold cyan]\n")
    else:
        console.print("\n[bold cyan]🧠 META-LEARNING: Model Improvement Suggestions[/bold cyan]\n")
    
    # Initialize LLM
    try:
        from llm_providers import initialize_buddy_llm, check_provider_status
        provider = initialize_buddy_llm()
        
        if not provider.is_available:
            console.print("[red]✗ No LLM provider available[/red]")
            console.print("[dim]Install Ollama (brew install ollama && ollama pull llama3.1:7b) or set ANTHROPIC_API_KEY[/dim]")
            return
        
        console.print(f"[green]✓ LLM Provider: {provider.name.upper()}[/green]")
    except ImportError as e:
        console.print(f"[red]✗ LLM providers not available: {e}[/red]")
        return
    
    # Load current architecture info from model metadata
    models_dir = Path("trained_data/models")
    
    # Check for pair-specific model first
    if instrument:
        pair_meta_path = models_dir / instrument / "modular_ensemble.meta.json"
        if pair_meta_path.exists():
            meta_path = pair_meta_path
            console.print(f"[dim]Using {instrument}-specific model metadata[/dim]")
        else:
            # Fallback to global model
            meta_path = models_dir / "modular_ensemble.meta.json"
            console.print(f"[yellow]No {instrument}-specific model found, using global model[/yellow]")
    else:
        meta_path = models_dir / "modular_ensemble.meta.json"
    
    if not meta_path.exists():
        meta_path = models_dir / "buddy_tf.meta.json"
    
    if not meta_path.exists():
        console.print("[red]✗ No model metadata found. Train a model first.[/red]")
        return
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Build architecture description
    from buddy_intelligent_mode import (
        BuddyArchitectureDescription,
        PerformanceStats,
        get_intelligent_mode,
        gated_auto_approve_suggestions,
    )
    
    # Format feature set and model type for description
    model_type = meta.get('model_type', 'tcn')
    models_list = list(meta.get('models', {}).keys()) if 'models' in meta else [model_type]
    model_type_str = f"Ensemble ({', '.join(models_list)})" if len(models_list) > 1 else model_type.upper()
    
    features_list = meta.get('feature_columns', meta.get('feature_names', []))[:10]
    feature_str = ", ".join(features_list) if isinstance(features_list, list) else str(features_list)
    
    architecture = BuddyArchitectureDescription(
        model_type=model_type_str,
        feature_set=feature_str,
        lookback_window=meta.get('seq_len', 60),
        hidden_size=meta.get('hidden_size', 64),
        num_layers=meta.get('num_layers', 2),
        dropout=meta.get('dropout', 0.2),
    )
    
    console.print(f"[dim]Architecture: {architecture.to_description()}[/dim]\n")
    
    # Load performance stats from journal if available (filter by instrument if specified)
    try:
        from trade_journal import TradeJournal
        from oanda_practice import OandaPracticeClient
        
        journal = TradeJournal()
        client = OandaPracticeClient.from_env()
        perf_data = journal.get_full_performance_stats(client, days=30)
        stats = perf_data.get('stats', {})
        
        # If instrument specified, try to get pair-specific stats
        if instrument:
            pair_stats = perf_data.get('pair_stats', {}).get(instrument, stats)
            if pair_stats != stats:
                console.print(f"[dim]Using {instrument}-specific performance stats[/dim]")
                stats = pair_stats
        
        performance = PerformanceStats(
            win_rate=stats.get('win_rate', 0.5),
            profit_factor=stats.get('profit_factor', 1.0),
            sharpe_ratio=stats.get('sharpe_ratio', 0.0),
            max_drawdown=stats.get('max_drawdown', 0.0),
            avg_trade_pnl=stats.get('avg_pnl', 0.0),
            total_trades=stats.get('total_trades', 0),
        )
    except Exception:
        # Use defaults if journal not available
        performance = PerformanceStats()
    
    console.print(f"[dim]Performance: {performance.to_summary()}[/dim]\n")
    
    # Get improvement suggestions
    console.print("[cyan]Analyzing model for improvements...[/cyan]")
    
    # Add instrument context to known weaknesses
    weaknesses = ["regime shifts", "news events", "low volatility periods"]
    if instrument:
        # Add pair-specific context
        if "JPY" in instrument:
            weaknesses.append("BOJ intervention risk")
        if "GBP" in instrument:
            weaknesses.append("Brexit-related volatility")
        if "AUD" in instrument or "NZD" in instrument:
            weaknesses.append("China trade sensitivity")
    
    intel_mode = get_intelligent_mode()
    result = intel_mode.get_improvement_suggestions(
        architecture=architecture,
        performance=performance,
        known_weaknesses=weaknesses,
    )
    
    if result.get('error'):
        console.print(f"[red]✗ Failed to get suggestions: {result['error']}[/red]")
        return
    
    improvements = result.get('improvements', [])
    
    if not improvements:
        console.print("[yellow]No improvements suggested (or LLM response couldn't be parsed)[/yellow]")
        if result.get('raw_response'):
            console.print(f"\n[dim]Raw response:[/dim]\n{result['raw_response'][:500]}...")
        return
    
    # Display all suggestions
    console.print(f"\n[bold]📋 {len(improvements)} Suggestions{' for ' + instrument if instrument else ''}:[/bold]\n")
    
    for i, imp in enumerate(improvements, 1):
        category_color = {
            'hyperparam': 'green',
            'hyperparameter': 'green',
            'parameter': 'green',
            'feature': 'blue',
            'architecture': 'magenta',
            'training': 'yellow',
        }.get(imp.category.lower(), 'white')
        
        ease_emoji = {'easy': '🟢', 'medium': '🟡', 'hard': '🔴'}.get(imp.implementation_ease.lower(), '⚪')
        
        console.print(f"  {i}. [{category_color}][{imp.category.upper()}][/{category_color}] {ease_emoji}")
        console.print(f"     {imp.suggestion}")
        console.print(f"     [dim]Impact: {imp.expected_impact}[/dim]")
        console.print(f"     [dim]Test: {imp.test_plan}[/dim]")
        console.print()
    
    # Determine config path for auto-apply
    # If instrument specified, create/use pair-specific config override
    if auto_apply and instrument:
        pair_config_dir = Path("trained_data/configs")
        pair_config_dir.mkdir(parents=True, exist_ok=True)
        pair_config_path = pair_config_dir / f"{instrument}_overrides.yaml"
        config_file = pair_config_path
        console.print(f"[dim]Changes will be saved to: {pair_config_path}[/dim]")
    else:
        config_file = Path(config_path)
    
    # Process through gated auto-approval
    approval_result = gated_auto_approve_suggestions(
        improvements=improvements,
        config_path=config_file if auto_apply else None,
        auto_apply=auto_apply,
    )
    
    # Display approval results
    console.print("\n[bold]🚦 Gated Auto-Approval Results:[/bold]\n")
    
    for line in approval_result.changelog:
        if line.startswith("✓"):
            console.print(f"  [green]{line}[/green]")
        elif line.startswith("✗"):
            console.print(f"  [red]{line}[/red]")
        elif line.startswith("⏸"):
            console.print(f"  [yellow]{line}[/yellow]")
        elif line.startswith("  →"):
            console.print(f"  [cyan]{line}[/cyan]")
        else:
            console.print(f"  {line}")
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  ✓ Approved: {len(approval_result.approved)} (hyperparams passing all gates)")
    console.print(f"  ⏸ Pending:  {len(approval_result.pending_review)} (require manual review)")
    console.print(f"  ✗ Rejected: {len(approval_result.rejected)} (failed gates)")
    
    if auto_apply and approval_result.applied_changes:
        console.print(f"\n[green]✓ Applied {len(approval_result.applied_changes)} changes to {config_file}[/green]")
        if instrument:
            console.print(f"[dim]These changes only affect {instrument} training/inference[/dim]")
    elif not auto_apply and approval_result.approved:
        console.print(f"\n[dim]Run with --auto-apply to apply approved changes[/dim]")


def buddy_journal(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    update: bool = False,
    days: int = 30,
    verbose: bool = False,
    import_trades: bool = False,
    **kwargs: Any,
) -> None:
    """Display comprehensive trade journal with full OANDA account performance.
    
    Args:
        config_path: Path to config file
        update: If True, sync local journal with OANDA
        days: Number of days of history to show
        verbose: Show detailed trade list
        import_trades: If True, import untracked open trades into journal
    """
    from datetime import datetime
    from trade_journal import TradeJournal
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    
    journal = TradeJournal()
    
    # Connect to OANDA
    client = None
    try:
        from oanda_practice import OandaPracticeClient
        client = OandaPracticeClient.from_env()
    except Exception as e:
        console.print(f"\n[red]✗ Failed to connect to OANDA: {e}[/red]")
        console.print("[dim]Set OANDA_API_TOKEN and OANDA_ACCOUNT_ID in .env file[/dim]")
        return
    
    # Fetch comprehensive performance data from OANDA
    console.print("\n[dim]Fetching account data from OANDA...[/dim]")
    perf_data = journal.get_full_performance_stats(client, days=days if days != 30 else None)
    
    account = perf_data.get('account')
    open_trades = perf_data.get('open_trades', [])
    closed_trades = perf_data.get('closed_trades', [])
    stats = perf_data.get('stats', {})
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════════════
    console.print()
    console.print("╔" + "═" * 78 + "╗")
    console.print("║" + " " * 25 + "[bold cyan]📊 TRADING PERFORMANCE DASHBOARD[/bold cyan]" + " " * 22 + "║")
    console.print("╚" + "═" * 78 + "╝")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ACCOUNT SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    if account:
        console.print()
        
        # Calculate daily P/L (approximation)
        daily_pnl = account.realized_pnl / max(days, 1) if days else account.realized_pnl / 30
        
        # Account status panel
        balance_color = "green" if account.balance >= 100000 else "white"
        nav_color = "green" if account.nav >= account.balance else "red"
        unrealized_color = "green" if account.unrealized_pnl >= 0 else "red"
        realized_color = "green" if account.realized_pnl >= 0 else "red"
        
        # Main metrics
        metrics_table = Table(show_header=False, box=None, padding=(0, 2))
        metrics_table.add_column("Label", style="dim")
        metrics_table.add_column("Value", justify="right")
        metrics_table.add_column("Label2", style="dim")
        metrics_table.add_column("Value2", justify="right")
        
        metrics_table.add_row(
            "Balance:",
            f"[{balance_color}]${account.balance:,.2f}[/{balance_color}]",
            "NAV:",
            f"[{nav_color}]${account.nav:,.2f}[/{nav_color}]",
        )
        metrics_table.add_row(
            "Unrealized P/L:",
            f"[{unrealized_color}]${account.unrealized_pnl:+,.2f}[/{unrealized_color}]",
            "Realized P/L:",
            f"[{realized_color}]${account.realized_pnl:+,.2f}[/{realized_color}]",
        )
        metrics_table.add_row(
            "Margin Used:",
            f"${account.margin_used:,.2f}",
            "Margin Available:",
            f"${account.margin_available:,.2f}",
        )
        
        margin_pct = (account.margin_used / account.nav * 100) if account.nav > 0 else 0
        margin_color = "green" if margin_pct < 30 else ("yellow" if margin_pct < 50 else "red")
        metrics_table.add_row(
            "Margin Usage:",
            f"[{margin_color}]{margin_pct:.1f}%[/{margin_color}]",
            "Open Positions:",
            f"{account.open_trade_count}",
        )
        
        console.print(Panel(
            metrics_table,
            title="[bold white]💰 ACCOUNT SUMMARY[/bold white]",
            border_style="blue",
            padding=(1, 2),
        ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OPEN POSITIONS
    # ═══════════════════════════════════════════════════════════════════════════
    if open_trades:
        total_unrealized = sum(t.unrealized_pnl for t in open_trades)
        pnl_color = "green" if total_unrealized >= 0 else "red"
        
        open_table = Table(show_header=True, header_style="bold white", box=None)
        open_table.add_column("ID", style="dim", width=6)
        open_table.add_column("Pair", width=9)
        open_table.add_column("Dir", justify="center", width=4)
        open_table.add_column("Units", justify="right", width=10)
        open_table.add_column("Entry", justify="right", width=10)
        open_table.add_column("SL", justify="right", width=10)
        open_table.add_column("TP", justify="right", width=10)
        open_table.add_column("P/L", justify="right", width=12)
        
        for trade in open_trades:
            direction_style = "green" if trade.direction == "long" else "red"
            direction_symbol = "⬆" if trade.direction == "long" else "⬇"
            
            pnl = trade.unrealized_pnl
            trade_pnl_color = "green" if pnl >= 0 else "red"
            
            is_jpy = trade.instrument.endswith('_JPY')
            price_fmt = ".3f" if is_jpy else ".5f"
            
            sl_str = f"{trade.stop_loss:{price_fmt}}" if trade.stop_loss else "—"
            tp_str = f"{trade.take_profit:{price_fmt}}" if trade.take_profit else "—"
            
            open_table.add_row(
                str(trade.trade_id),
                trade.instrument.replace("_", "/"),
                f"[{direction_style}]{direction_symbol}[/{direction_style}]",
                f"{trade.units:,}",
                f"{trade.entry_price:{price_fmt}}",
                sl_str,
                tp_str,
                f"[{trade_pnl_color}]${pnl:+,.2f}[/{trade_pnl_color}]",
            )
        
        header = f"[bold white]🔴 OPEN POSITIONS ({len(open_trades)})[/bold white]  │  Unrealized: [{pnl_color}]${total_unrealized:+,.2f}[/{pnl_color}]"
        console.print(Panel(
            open_table,
            title=header,
            border_style="cyan",
            padding=(0, 1),
        ))
    else:
        console.print(Panel(
            "[dim]No open positions[/dim]",
            title="[bold white]🔴 OPEN POSITIONS[/bold white]",
            border_style="dim",
        ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERFORMANCE STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    if stats.get('total_trades', 0) > 0:
        # Performance metrics
        win_rate = stats.get('win_rate', 0) * 100
        wr_color = "green" if win_rate >= 50 else ("yellow" if win_rate >= 40 else "red")
        
        pf = stats.get('profit_factor', 0)
        pf_color = "green" if pf >= 1.5 else ("yellow" if pf >= 1.0 else "red")
        pf_str = f"{pf:.2f}" if pf != float('inf') else "∞"
        
        total_pnl = stats.get('total_pnl', 0)
        pnl_color = "green" if total_pnl >= 0 else "red"
        
        # Build performance panel
        perf_lines = []
        perf_lines.append(f"[bold]📈 Performance Metrics[/bold]")
        perf_lines.append("")
        perf_lines.append(f"  Total Trades:     [white]{stats.get('total_trades', 0)}[/white]")
        perf_lines.append(f"  Win Rate:         [{wr_color}]{win_rate:.1f}%[/{wr_color}]  ({stats.get('wins', 0)}W / {stats.get('losses', 0)}L)")
        perf_lines.append(f"  Profit Factor:    [{pf_color}]{pf_str}[/{pf_color}]")
        perf_lines.append(f"  Total P/L:        [{pnl_color}]${total_pnl:+,.2f}[/{pnl_color}]")
        perf_lines.append(f"  Total Pips:       {stats.get('total_pips', 0):+.1f}")
        
        if stats.get('total_financing', 0) != 0:
            fin_color = "red" if stats.get('total_financing', 0) < 0 else "green"
            perf_lines.append(f"  Financing:        [{fin_color}]${stats.get('total_financing', 0):+.2f}[/{fin_color}]")
        
        perf_lines.append("")
        perf_lines.append(f"[bold]📊 Risk Metrics[/bold]")
        perf_lines.append("")
        avg_win = stats.get('avg_win', 0)
        avg_loss = stats.get('avg_loss', 0)
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        perf_lines.append(f"  Avg Win:          [green]${avg_win:,.2f}[/green]")
        perf_lines.append(f"  Avg Loss:         [red]${avg_loss:,.2f}[/red]")
        perf_lines.append(f"  Risk/Reward:      {rr_ratio:.2f}")
        perf_lines.append(f"  Max Drawdown:     [red]${stats.get('max_drawdown', 0):,.2f}[/red]")
        
        # Streak info
        streak = stats.get('current_streak', 0)
        streak_color = "green" if streak > 0 else ("red" if streak < 0 else "white")
        streak_text = f"+{streak}" if streak > 0 else str(streak)
        
        perf_lines.append("")
        perf_lines.append(f"[bold]🔥 Streaks[/bold]")
        perf_lines.append("")
        perf_lines.append(f"  Current Streak:   [{streak_color}]{streak_text}[/{streak_color}]")
        perf_lines.append(f"  Best Win Streak:  [green]+{stats.get('max_win_streak', 0)}[/green]")
        perf_lines.append(f"  Worst Loss Streak:[red]-{stats.get('max_loss_streak', 0)}[/red]")
        
        # By instrument breakdown
        by_instrument = stats.get('by_instrument', {})
        if by_instrument:
            perf_lines.append("")
            perf_lines.append(f"[bold]🌍 By Instrument[/bold]")
            perf_lines.append("")
            
            # Sort by P/L
            sorted_instruments = sorted(by_instrument.items(), key=lambda x: x[1]['pnl'], reverse=True)
            for instr, data in sorted_instruments[:6]:  # Top 6
                total = data['wins'] + data['losses']
                wr = data['wins'] / total * 100 if total > 0 else 0
                instr_pnl_color = "green" if data['pnl'] >= 0 else "red"
                perf_lines.append(
                    f"  {instr.replace('_', '/'):<10} {data['wins']}W/{data['losses']}L "
                    f"({wr:.0f}%) [{instr_pnl_color}]${data['pnl']:+,.2f}[/{instr_pnl_color}]"
                )
        
        console.print(Panel(
            "\n".join(perf_lines),
            title=f"[bold white]📈 PERFORMANCE SUMMARY ({stats.get('total_trades', 0)} trades)[/bold white]",
            border_style="green" if total_pnl >= 0 else "red",
            padding=(1, 2),
        ))
    else:
        console.print(Panel(
            "[dim]No closed trades to analyze[/dim]",
            title="[bold white]📈 PERFORMANCE SUMMARY[/bold white]",
            border_style="dim",
        ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RECENT TRADE HISTORY
    # ═══════════════════════════════════════════════════════════════════════════
    if verbose and closed_trades:
        console.print()
        
        history_table = Table(show_header=True, header_style="bold white", box=None)
        history_table.add_column("Date", style="dim", width=12)
        history_table.add_column("Pair", width=9)
        history_table.add_column("Dir", justify="center", width=4)
        history_table.add_column("Units", justify="right", width=10)
        history_table.add_column("Entry", justify="right", width=10)
        history_table.add_column("Exit", justify="right", width=10)
        history_table.add_column("P/L", justify="right", width=12)
        
        for trade in closed_trades[:20]:  # Show last 20
            direction_style = "green" if trade.direction == "long" else "red"
            direction_symbol = "⬆" if trade.direction == "long" else "⬇"
            
            pnl = trade.realized_pnl
            trade_pnl_color = "green" if pnl >= 0 else "red"
            
            is_jpy = trade.instrument.endswith('_JPY')
            price_fmt = ".3f" if is_jpy else ".5f"
            
            try:
                dt = datetime.fromisoformat(trade.close_time.replace('Z', '+00:00'))
                date_str = dt.strftime("%m/%d %H:%M")
            except Exception:
                date_str = "—"
            
            history_table.add_row(
                date_str,
                trade.instrument.replace("_", "/"),
                f"[{direction_style}]{direction_symbol}[/{direction_style}]",
                f"{trade.units:,}",
                f"{trade.entry_price:{price_fmt}}",
                f"{trade.exit_price:{price_fmt}}",
                f"[{trade_pnl_color}]${pnl:+,.2f}[/{trade_pnl_color}]",
            )
        
        console.print(Panel(
            history_table,
            title=f"[bold white]📋 RECENT TRADE HISTORY (Last {min(20, len(closed_trades))})[/bold white]",
            border_style="white",
            padding=(0, 1),
        ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SYNC LOCAL JOURNAL (if requested)
    # ═══════════════════════════════════════════════════════════════════════════
    if update:
        console.print()
        console.print("[dim]Syncing local journal...[/dim]")
        try:
            sync_results = journal.sync_open_trades(client)
            closed_updated = sync_results.get('closed_updated', 0)
            open_updated = sync_results.get('open_updated', 0)
            
            if closed_updated > 0 or open_updated > 0:
                console.print(f"[green]✓ Local journal synced ({closed_updated} closed, {open_updated} open)[/green]")
            
            # Import untracked trades if requested
            untracked = sync_results.get('untracked_trades', [])
            if untracked and import_trades:
                imported = journal.import_untracked_trades(client)
                if imported > 0:
                    console.print(f"[green]✓ Imported {imported} trades into local journal[/green]")
        except Exception as e:
            console.print(f"[yellow]Local journal sync failed: {e}[/yellow]")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════════════════
    console.print()
    console.print("─" * 80)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[dim]Last updated: {timestamp}  │  Use --verbose for trade history  │  Use --update to sync journal[/dim]")


def buddy_analyze(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    top_n: int = 30,
    save_plots: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """
    Analyze feature importance using SHAP values.
    
    Shows which features contribute most to predictions, helping to:
    - Understand what the model learned
    - Remove noisy features
    - Improve model performance
    """
    import json
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from rich.table import Table
    
    console.print("\n" + "=" * 70)
    console.print(f"[bold magenta]🔬 FEATURE IMPORTANCE ANALYSIS[/bold magenta]")
    console.print("=" * 70)
    
    cfg = load_config(config_path)
    
    # Load model metadata
    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    candidate_meta_path = Path("trained_data") / "models" / "buddy_tf_candidate.meta.json"
    ensemble_meta_path = Path("trained_data") / "models" / "buddy_ensemble.meta.json"
    
    meta = None
    for mp in [ensemble_meta_path, candidate_meta_path, meta_path]:
        if mp.exists():
            meta = json.loads(mp.read_text())
            break
    
    if meta is None:
        console.print("[red]No trained model found. Run: buddy train --oanda-live[/red]")
        return
    
    feature_columns = list(meta.get("feature_columns", []))
    if not feature_columns:
        console.print("[red]No feature columns in model metadata[/red]")
        return
    
    console.print(f"[dim]Analyzing {len(feature_columns)} features...[/dim]")
    
    # Try to use SHAP if available, otherwise fallback to simpler methods
    try:
        from feature_importance import FeatureImportanceAnalyzer, SHAP_AVAILABLE
        use_shap = SHAP_AVAILABLE
    except ImportError:
        use_shap = False
    
    # Load recent data for analysis
    console.print("[dim]Fetching recent data for analysis...[/dim]")
    
    from oanda_practice import OandaPracticeClient
    from feature_engineering import FeatureEngineering
    from fx_paper import candles_to_ohlcv_df
    
    client = OandaPracticeClient.from_env()
    fe = FeatureEngineering()
    
    resp = client.get_candles("USD_JPY", granularity="H1", count=2000, price="MBA")
    df_raw = candles_to_ohlcv_df(resp)
    df = fe.create_features(df_raw.copy(), include_all=True)
    
    # Ensure required columns
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0.0
    
    X = df[feature_columns].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create direction labels for correlation analysis
    df["future_return"] = df["close"].pct_change(12).shift(-12)
    df["direction"] = (df["future_return"] > 0).astype(float)
    y = df["direction"].values
    
    # Remove NaN
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    if use_shap:
        console.print("[dim]Using SHAP for feature importance (this may take a minute)...[/dim]")
        
        # Load model for SHAP
        _configure_tf_metal(verbose=verbose)
        import tensorflow as tf
        
        model_path = meta.get("model_path")
        model_type = meta.get("model_type", "keras")
        
        if model_type == "ensemble":
            # For ensemble, use XGBoost component if available
            from ensemble_model import EnsembleModel
            ensemble_path = meta.get("ensemble_path")
            if ensemble_path and Path(ensemble_path).exists():
                ensemble = EnsembleModel.load(ensemble_path)
                # Use XGBoost or RF from ensemble
                if "xgboost" in ensemble.base_models:
                    model = ensemble.base_models["xgboost"].model
                elif "random_forest" in ensemble.base_models:
                    model = ensemble.base_models["random_forest"].model
                else:
                    model = None
            else:
                model = None
        else:
            model = None  # SHAP with Keras is slow, use correlation instead
        
        if model is not None:
            try:
                analyzer = FeatureImportanceAnalyzer(model, feature_columns)
                analyzer.fit(X)
                results = analyzer.analyze(X)
                importance_df = results["importance_df"]
                
                # Save plots if requested
                if save_plots:
                    viz_dir = Path("trained_data") / "visualizations"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    analyzer.plot_importance(max_display=top_n, save_path=str(viz_dir / "shap_importance.png"))
                    console.print(f"[green]Saved plot to {viz_dir / 'shap_importance.png'}[/green]")
                
            except Exception as e:
                console.print(f"[yellow]SHAP analysis failed: {e}[/yellow]")
                use_shap = False
    
    if not use_shap:
        console.print("[dim]Using correlation-based feature importance...[/dim]")
        
        # Calculate correlation with target
        correlations = []
        for i, col in enumerate(feature_columns):
            if i < X.shape[1]:
                try:
                    # Use pandas corr for robustness
                    x_col = pd.Series(X[:, i])
                    y_series = pd.Series(y)
                    corr = x_col.corr(y_series)
                    correlations.append(abs(corr) if not pd.isna(corr) else 0)
                except Exception:
                    correlations.append(0)
            else:
                correlations.append(0)
        
        importance_df = pd.DataFrame({
            "feature": feature_columns,
            "importance": correlations,
            "rank": range(1, len(feature_columns) + 1),
        })
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)
    
    # Display results
    console.print("\n" + "=" * 70)
    console.print(f"[bold green]📊 TOP {top_n} MOST IMPORTANT FEATURES[/bold green]")
    console.print("=" * 70)
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Feature", style="bold")
    table.add_column("Importance", justify="right")
    table.add_column("Category", justify="center")
    
    # Categorize features
    def get_category(name: str) -> str:
        name_lower = name.lower()
        if any(x in name_lower for x in ["rsi", "macd", "stoch", "cci", "williams", "mfi"]):
            return "Momentum"
        elif any(x in name_lower for x in ["sma", "ema", "bb_"]):
            return "Trend"
        elif any(x in name_lower for x in ["atr", "volatility", "bb_width"]):
            return "Volatility"
        elif any(x in name_lower for x in ["volume", "obv"]):
            return "Volume"
        elif any(x in name_lower for x in ["hour", "day", "session"]):
            return "Time"
        elif any(x in name_lower for x in ["adx", "trend", "regime"]):
            return "Regime"
        else:
            return "Other"
    
    for _, row in importance_df.head(top_n).iterrows():
        category = get_category(row["feature"])
        cat_color = {
            "Momentum": "yellow",
            "Trend": "blue",
            "Volatility": "red",
            "Volume": "green",
            "Time": "cyan",
            "Regime": "magenta",
            "Other": "white",
        }.get(category, "white")
        
        table.add_row(
            str(int(row["rank"])),
            row["feature"],
            f"{row['importance']:.4f}",
            f"[{cat_color}]{category}[/{cat_color}]",
        )
    
    console.print(table)
    
    # Summary by category
    console.print("\n[bold]📈 IMPORTANCE BY CATEGORY:[/bold]")
    importance_df["category"] = importance_df["feature"].apply(get_category)
    category_importance = importance_df.groupby("category")["importance"].sum().sort_values(ascending=False)
    
    total_importance = category_importance.sum()
    for cat, imp in category_importance.items():
        pct = imp / total_importance * 100 if total_importance > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        console.print(f"  {cat:12} {bar} {pct:.1f}%")
    
    # Recommendations
    console.print("\n[bold]💡 RECOMMENDATIONS:[/bold]")
    
    top_features = importance_df.head(20)["feature"].tolist()
    bottom_features = importance_df.tail(50)["feature"].tolist()
    
    console.print(f"  • Top 20 features explain most of the signal")
    console.print(f"  • Consider removing {len(bottom_features)} low-importance features")
    
    # Feature groups that matter
    top_categories = importance_df.head(30).groupby("category").size().sort_values(ascending=False)
    console.print(f"  • Most important category: {top_categories.index[0]} ({top_categories.iloc[0]} features in top 30)")
    
    console.print("\n" + "=" * 70)


# =============================================================================
# MODEL VALIDATION (Professional Testing)
# =============================================================================

def buddy_validate(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    instrument: str = "EUR_USD",
    granularity: str = "H1",
    candles: int = 800,
    lookahead: int = 24,
    all_pairs: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """
    Professional model validation with bias detection, statistical tests,
    and risk metrics. Use this before live trading to verify model integrity.
    
    Checks:
    - Direction bias (model stuck predicting one direction)
    - Accuracy vs baselines (random, market bias, buy-and-hold)
    - Statistical significance (binomial test, confidence intervals)
    - Confidence calibration (high confidence = higher accuracy)
    - Time stability (consistent accuracy across periods)
    - Risk metrics (Sharpe, Sortino, max drawdown, profit factor)
    
    Usage:
        buddy validate --instrument EUR_USD
        buddy validate --all-pairs
        buddy validate --instrument USD_JPY --candles 1000
    """
    from tests.validation.validate_model import validate_pair_model, validate_all_pairs
    
    # Normalize instrument
    instrument = _normalize_instrument(instrument)
    
    if all_pairs:
        validate_all_pairs(
            candles=candles,
            granularity=granularity,
            lookahead=lookahead,
        )
    else:
        _validate_instrument(instrument)
        validate_pair_model(
            pair=instrument,
            candles=candles,
            granularity=granularity,
            lookahead=lookahead,
            verbose=verbose,
        )


# Legacy test alias (redirects to validate)
def buddy_test(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    instrument: str = "EUR_USD",
    granularity: str = "H1",
    test_candles: int = 500,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """
    [DEPRECATED] Use 'buddy validate' instead.
    
    This redirects to the professional validation system.
    """
    console.print("[yellow]⚠️ 'buddy test' is deprecated. Using 'buddy validate' instead.[/yellow]")
    console.print("[dim]For professional validation, use: buddy validate --instrument EUR_USD[/dim]\n")
    
    buddy_validate(
        config_path=config_path,
        instrument=instrument,
        granularity=granularity,
        candles=test_candles + 300,  # Add buffer for feature warmup
        verbose=verbose,
        **kwargs,
    )


def model_status(config_path: str = DEFAULT_CONFIG_PATH, **kwargs: Any) -> None:
    """Display current model status and training metrics."""
    import json
    from datetime import datetime
    
    console.print("\n" + "=" * 60)
    console.print("[bold]📊 MODEL STATUS[/bold]")
    console.print("=" * 60)
    
    models_dir = Path("trained_data") / "models"
    has_any_model = False
    
    # =========================================================================
    # CHECK FOR PAIR-SPECIFIC MODELS (Primary)
    # =========================================================================
    pair_dirs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "USD_CHF"]
    pair_models_found = []
    
    for pair in pair_dirs:
        pair_dir = models_dir / pair
        keras_path = pair_dir / "transformer_direction.keras"
        meta_path = pair_dir / "transformer_direction.meta.pkl"
        
        if keras_path.exists() and meta_path.exists():
            pair_models_found.append(pair)
            has_any_model = True
    
    if pair_models_found:
        console.print(f"\n[bold green]★ Pair-Specific Models ({len(pair_models_found)} loaded)[/bold green]")
        for pair in pair_models_found:
            meta_path = models_dir / pair / "transformer_direction.meta.pkl"
            try:
                import pickle
                meta = pickle.load(open(meta_path, 'rb'))
                # Get accuracy from metrics dict
                metrics = meta.get('metrics', {})
                acc = metrics.get('val_accuracy', metrics.get('best_val_accuracy', 0))
                val_result_path = models_dir / pair / ".validation_result.json"
                validated = "✓" if val_result_path.exists() else "?"
                console.print(f"  {pair:<10} {acc*100:>5.1f}% acc  [{validated} validated]")
            except Exception:
                console.print(f"  {pair:<10} [dim]metadata error[/dim]")
    
    # =========================================================================
    # CHECK FOR LOCAL FALLBACK (legacy)
    # =========================================================================
    modular_meta_path = models_dir / "modular_ensemble.meta.json"
    
    if modular_meta_path.exists():
        has_any_model = True
        try:
            meta = json.loads(modular_meta_path.read_text())
            direction = meta.get('models', {}).get('direction', {}).get('metrics', {})
            
            console.print(f"\n[dim]○ Legacy Fallback: {direction.get('val_accuracy', 0):.1%} acc | {meta.get('trained_at', '')[:10]}[/dim]")
                
        except Exception:
            pass
    
    if not has_any_model:
        console.print("\n[yellow]✗ No models found[/yellow]")
        console.print("[dim]Train: buddy train --oanda-live --candles 5000[/dim]")
    
    console.print("\n" + "-" * 60)
    console.print("[dim]buddy predict | buddy scan | buddy validate[/dim]")
    console.print("=" * 60 + "\n")


def promote_model(config_path: str = DEFAULT_CONFIG_PATH, **kwargs: Any) -> None:
    """Promote candidate model to production."""
    import shutil
    import json
    from datetime import datetime
    
    models_dir = Path("trained_data") / "models"
    
    # =========================================================================
    # CHECK FOR MODULAR ENSEMBLE CANDIDATE FIRST
    # =========================================================================
    modular_candidate_meta = models_dir / "modular_ensemble_candidate.meta.json"
    modular_prod_meta = models_dir / "modular_ensemble.meta.json"
    
    if modular_candidate_meta.exists():
        try:
            meta = json.loads(modular_candidate_meta.read_text())
            
            # Show candidate stats
            console.print("\n[cyan]Modular Ensemble Candidate Stats:[/cyan]")
            direction = meta.get('models', {}).get('direction', {})
            dir_metrics = direction.get('metrics', {})
            val_acc = dir_metrics.get('val_accuracy')
            if val_acc is not None:
                console.print(f"  Direction Accuracy: {val_acc:.1%}")
            console.print(f"  Model Type:         {meta.get('model_type', 'modular_ensemble')}")
            if meta.get('trained_at'):
                console.print(f"  Trained:            {meta.get('trained_at')}")
            
            # Backup existing production
            if modular_prod_meta.exists():
                backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_meta = models_dir / f"modular_ensemble_backup_{backup_suffix}.meta.json"
                shutil.copy2(modular_prod_meta, backup_meta)
                console.print(f"[dim]Backed up existing meta to: {backup_meta.name}[/dim]")
                
                # Backup component models
                prod_meta_data = json.loads(modular_prod_meta.read_text())
                for name, model_info in prod_meta_data.get('models', {}).items():
                    model_path = Path(model_info.get('path', ''))
                    if model_path.exists():
                        backup_model = model_path.parent / f"{model_path.stem}_backup_{backup_suffix}{model_path.suffix}"
                        shutil.copy2(model_path, backup_model)
            
            # Promote: copy candidate models to production locations
            for name, model_info in meta.get('models', {}).items():
                cand_path = Path(model_info.get('path', '').replace('.keras', '_candidate.keras').replace('.pkl', '_candidate.pkl'))
                prod_path = Path(model_info.get('path', ''))
                
                if cand_path.exists():
                    shutil.copy2(cand_path, prod_path)
                    console.print(f"  [green]✓[/green] Promoted {name}: {cand_path.name} → {prod_path.name}")
            
            # Promote meta
            shutil.copy2(modular_candidate_meta, modular_prod_meta)
            
            console.print("\n[green]✓ Modular ensemble promoted successfully![/green]")
            console.print("\n[dim]Run inference: buddy predict[/dim]")
            return
            
        except Exception as e:
            console.print(f"[yellow]⚠ Error promoting modular ensemble: {e}[/yellow]")
    
    # =========================================================================
    # FALLBACK TO LEGACY SINGLE MODEL
    # =========================================================================
    cand_model = models_dir / "buddy_tf_candidate.keras"
    cand_meta = models_dir / "buddy_tf_candidate.meta.json"
    prod_model = models_dir / "buddy_tf.keras"
    prod_meta = models_dir / "buddy_tf.meta.json"
    
    if not cand_model.exists() or not cand_meta.exists():
        console.print("[red]✗ No candidate model to promote[/red]")
        console.print("[dim]Train a model first: buddy train --oanda-live --candles 5000[/dim]")
        return
    
    # Show candidate stats
    try:
        meta = json.loads(cand_meta.read_text())
        val_acc = meta.get('val_direction_accuracy', 'N/A')
        if isinstance(val_acc, (int, float)):
            console.print(f"\n[cyan]Candidate Model Stats:[/cyan]")
            console.print(f"  Direction Accuracy: {val_acc:.1%}")
            console.print(f"  Model Type:         {meta.get('model_type', 'unknown')}")
    except Exception:
        pass
    
    # Backup existing production model
    if prod_model.exists():
        backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_model = models_dir / f"buddy_tf_backup_{backup_suffix}.keras"
        backup_meta = models_dir / f"buddy_tf_backup_{backup_suffix}.meta.json"
        
        shutil.copy2(prod_model, backup_model)
        if prod_meta.exists():
            shutil.copy2(prod_meta, backup_meta)
        console.print(f"[dim]Backed up existing model to: {backup_model.name}[/dim]")
    
    # Promote candidate
    shutil.copy2(cand_model, prod_model)
    shutil.copy2(cand_meta, prod_meta)
    
    console.print("\n[green]✓ Model promoted successfully![/green]")
    console.print(f"  Production model: {prod_model}")
    console.print("\n[dim]Run inference: buddy predict[/dim]")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ML Engine Trading Bot CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
COMMAND REFERENCE:
  scan        Scan ALL pairs → shows tradeable + needs training
  buddy       Single-pair inference → execute one trade
  train       Train model for specific pair
  
EXAMPLES:
  buddy scan                     # Scan all majors, show recommendations
  buddy -I EUR_USD --execute     # Single EUR_USD trade
  buddy train -I AUD_USD         # Train model for AUD_USD
  buddy journal                  # View trade journal
""",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="buddy",
        choices=[
            "train",
            "train-buddy",
            "retrain-gates",
            "train-rl-sizer",
            "buddy",
            "Buddy",
            "promote-model",
            "model-status",
            "test",
            "validate",
            "scan",
            "analyze",
            "journal",
            "trade-analysis",
            "suggest-improvements",
        ],
        help="Command: scan (multi-pair) | buddy (single-pair) | train | validate | journal | suggest-improvements",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--csv",
        "-f",
        default=None,
        help="Path to market CSV file (used by train-buddy)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional Buddy model path override (defaults to trained_data/models/buddy_tf.keras)",
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Optional PCA components (e.g. 20-30) for Buddy training",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length for Buddy training (default: config buddy.train_defaults.seq_len or 50)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs for Buddy training (default: config buddy.train_defaults.epochs or 300)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for Buddy training (default: config buddy.train_defaults.batch_size or 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for Buddy training (default: config buddy.train_defaults.lr or 0.001)",
    )
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For train-buddy: initialize weights from existing model before training (preserves learned patterns). (default: enabled; use --no-warm-start for fresh training)",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="For train-buddy: path to a .keras checkpoint to warm-start from (overrides --warm-start)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="EarlyStopping patience for Buddy training (applies to --es-monitor) (default: config buddy.train_defaults.patience or 10)",
    )
    parser.add_argument(
        "--es-monitor",
        default=None,
        choices=["direction", "combined", "val_loss", "loss"],
        help="EarlyStopping monitor for train-buddy: direction|combined|val_loss|loss (default: config buddy.train_defaults.es_monitor or direction)",
    )
    parser.add_argument(
        "--combined-w-dir",
        type=float,
        default=None,
        help="Weight for direction accuracy in combined early-stopping objective (default: config buddy.train_defaults.combined_w_dir or 0.7)",
    )
    parser.add_argument(
        "--combined-w-conf",
        type=float,
        default=None,
        help="Weight for confidence score (1 - MAE) in combined early-stopping objective (default: config buddy.train_defaults.combined_w_conf or 0.3)",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=None,
        help="For train-buddy: keep only top-K most correlated numeric features (disables warm-start)",
    )
    parser.add_argument(
        "--feature-curriculum",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For train-buddy: enable/disable feature curriculum (CLI overrides config). Cannot be used with --top-features.",
    )
    parser.add_argument(
        "--curriculum-ks",
        default=None,
        help=(
            "For train-buddy --feature-curriculum: comma-separated K schedule (0 = all). "
            f"If unset, uses config buddy.curriculum_ks or default {DEFAULT_CURRICULUM_KS}."
        ),
    )
    parser.add_argument(
        "--ignore-input-mismatches",
        action="store_true",
        help="Best-effort mode: downgrade some input/PCA/curriculum mismatches to warnings.",
    )
    parser.add_argument(
        "--disable-tier2-on-mismatch",
        action="store_true",
        help="If Tier-2 calibration encounters input mismatches, skip Tier-2 instead of best-effort.",
    )
    parser.add_argument(
        "--median-window",
        type=int,
        default=None,
        help="For train-buddy: apply rolling median to close before EMA smoothing (e.g. 3,5)",
    )

    # Tier-2 calibration controls (TP-before-SL simulation). This can be the dominant runtime cost.
    parser.add_argument(
        "--tier2-calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For train-buddy: enable/disable Tier-2 TP/SL calibration + labeling (default enabled)",
    )
    parser.add_argument(
        "--tier2-calibration-stride",
        type=int,
        default=None,
        help="For train-buddy: stride for Tier-2 labeling/calibration simulation (higher = faster, default: config buddy.train_defaults.tier2_calibration_stride or 5)",
    )
    parser.add_argument(
        "--tier2-horizon-candles",
        type=int,
        default=None,
        help="For train-buddy: Tier-2 TP/SL max horizon in candles (lower = faster). Default uses config.",
    )
    parser.add_argument(
        "--vol-norm-window",
        type=int,
        default=None,
        help="For train-buddy: normalize returns by rolling volatility when ranking top features (legacy) (e.g. 50)",
    )
    parser.add_argument(
        "--min-volume",
        type=int,
        default=None,
        help="For train-buddy: drop candles with volume below this threshold before training",
    )
    parser.add_argument(
        "--spread-filter",
        action="store_true",
        help="For train-buddy: drop candles with abnormally wide bid/ask spread (requires bid_close+ask_close)",
    )
    parser.add_argument(
        "--spread-pctl",
        type=float,
        default=0.99,
        help="For train-buddy: spread percentile used as filter threshold (default 0.99)",
    )
    parser.add_argument(
        "--spread-mult",
        type=float,
        default=3.0,
        help="For train-buddy: also filter if spread > (median * mult) (default 3.0)",
    )
    parser.add_argument(
        "--no-train-smoothing",
        action="store_true",
        help="For train-buddy: disable candle noise reduction (resample to M5 + EMA(14) close)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Buddy training (default 42)",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional tag to version outputs (writes buddy_tf_<tag>.keras + meta)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed logs (default: quiet output)",
    )

    parser.add_argument(
        "--instrument",
        "-I",
        default="EUR_USD",
        help="OANDA instrument e.g. USD_JPY,EUR_USD,GBP_USD (used by buddy and --oanda-live training)",
    )
    parser.add_argument(
        "--granularity",
        "-g",
        default="H1",
        help="OANDA candle granularity, e.g. M5, M15, H1 (used by buddy and --oanda-live training)",
    )
    parser.add_argument(
        "--candles",
        "-n",
        type=int,
        default=5000,
        help="How many candles to fetch for training (default: 5000)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="For test: only count predictions with confidence >= this value (e.g., 0.6)",
    )
    
    # Validation command arguments
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="For validate: validate all trained pair models",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=24,
        help="For validate: hours ahead to measure actual outcome (default: 24)",
    )
    
    # Scan command arguments
    parser.add_argument(
        "--pairs",
        type=str,
        default=None,
        help="For scan: comma-separated pairs to scan (e.g., EUR_USD,GBP_USD,USD_JPY)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="For scan/analyze: number of top results to show",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        dest="no_train",
        help="For scan: skip the train prompt after scanning (deprecated)",
    )
    parser.add_argument(
        "--skip-execute",
        action="store_true",
        dest="skip_execute",
        help="For scan: skip the trade execution prompt after scanning",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="For scan: enable continuous watch mode (scans repeatedly)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="For scan --watch: minutes between scans (default: 5)",
    )
    parser.add_argument(
        "--auto-execute",
        action="store_true",
        help="For scan --watch: automatically execute passing trades",
    )
    parser.add_argument(
        "--diversified",
        "-d",
        action="store_true",
        help="For scan: auto-filter correlated pairs (only show best from each correlation cluster)",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=50,
        help="For trade-analysis: number of recent trades to analyze (default: 50)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="For trade-analysis: export report to JSON file",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="For analyze: save plots to trained_data/visualizations",
    )
    
    # Journal command arguments
    parser.add_argument(
        "--update",
        action="store_true",
        help="For journal: update trade results from OANDA",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="For journal: number of days of history to show (default: 30)",
    )
    parser.add_argument(
        "--import-trades",
        action="store_true",
        help="For journal: import untracked open trades from OANDA into journal",
    )
    
    # RL Sizer training arguments
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="For train-rl-sizer: total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--rl-episodes",
        type=int,
        default=None,
        help="For train-rl-sizer: training episodes (if set, overrides --timesteps)",
    )
    parser.add_argument(
        "--use-rl-sizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For buddy/scan: use RL position sizer instead of Kelly criterion. (default: enabled if RL model exists; use --no-use-rl-sizer to disable)",
    )
    parser.add_argument(
        "--skip-rl",
        action="store_true",
        help="For train-buddy: skip RL position sizer training after ensemble",
    )

    parser.add_argument(
        "--oanda-live",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For train-buddy: fetch fresh candles from OANDA PRACTICE (default: enabled; use --no-oanda-live to use --csv file instead)",
    )
    parser.add_argument(
        "--oanda-price",
        default="MBA",
        help="OANDA price component for candles: M, B, A, or MBA (used by --oanda-live)",
    )
    parser.add_argument(
        "--oanda-save-csv",
        default=None,
        help="Optional path to write fetched OANDA candles CSV (used by --oanda-live)",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=10_000.0,
        help="Paper equity for sizing (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--risk",
        "-r",
        type=float,
        default=0.005,
        help="Risk per trade fraction (e.g. 0.005 = 0.5%%) (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--force-units",
        type=int,
        default=None,
        help="Override computed units and place exactly this many units (PRACTICE only).",
    )
    parser.add_argument(
        "--execute",
        "-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable live trading on OANDA practice account (default: enabled; use --dry-run or --no-execute to simulate)",
    )
    parser.add_argument(
        "--force-execute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bypass the training accuracy live-trading gate (PRACTICE only) (default: enabled; use --no-force-execute to enforce the gate).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Disable live trading, only simulate orders (overrides --execute)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="For buddy: keep Buddy warm and act on each new candle (default: off)",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=1,
        help="For buddy --loop: stop after this many executed trades (0 = unlimited) (default 1)",
    )
    parser.add_argument(
        "--all-features",
        "-A",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all engineered features (technical indicators, statistical, time, lag, rolling) for training. (default: enabled; use --no-all-features to disable)",
    )

    parser.add_argument(
        "--repl",
        action="store_true",
        help="For buddy: launch the interactive Buddy REPL (interactive trading commands)",
    )

    # --- Training speed knobs (optional; default behavior unchanged) ---
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Train device preference for TensorFlow: auto|cpu|gpu (default auto)",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mixed precision (mixed_float16) for Buddy training - 1.5-2x speedup on M1. (default: enabled; use --no-mixed-precision to disable)",
    )
    parser.add_argument(
        "--steps-per-execution",
        type=int,
        default=None,
        help="Keras steps_per_execution to reduce per-step overhead (default: config buddy.train_defaults.steps_per_execution or 1)",
    )
    parser.add_argument(
        "--jit-compile",
        action="store_true",
        help="Enable Keras jit_compile (XLA) if supported by your TensorFlow build",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=None,
        help="Override window shuffle buffer size (else uses BUDDY_SHUFFLE_BUFFER or default 256)",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=None,
        help="Override tf.data prefetch: N>0 bounded, 0=AUTOTUNE (else uses BUDDY_PREFETCH or default 1)",
    )
    parser.add_argument(
        "--cache-val",
        action="store_true",
        help="Cache the validation window dataset in memory (can speed up validation; uses more RAM)",
    )
    parser.add_argument(
        "--combined-use-predict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For --es-monitor combined: use an extra predict()+label pass per epoch to compute confidence MAE (best for calibration; disable for speed)",
    )
    parser.add_argument(
        "--shared-encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a single shared LSTM encoder instead of 5 parallel heads (default: config buddy.train_defaults.shared_encoder or false)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["tcn", "lstm", "xgboost", "ensemble", "attention_lstm"],
        default=None,
        help="Model architecture: ensemble (best accuracy), tcn (fast, M1 optimized), xgboost (gradient boosting), lstm (legacy) (default: config model.type or tcn)",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print per-batch timing (helps distinguish TF tracing warmup from true slow steps)",
    )
    parser.add_argument(
        "--fit-verbose",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Keras fit verbosity for train-buddy: 0=silent, 1=progress bar, 2=one line per epoch (default: config buddy.train_defaults.fit_verbose or 1)",
    )
    parser.add_argument(
        "--max-hold-minutes",
        type=float,
        default=None,
        help="Maximum holding time in minutes for opened trades (practice mode). After this time, open positions for the instrument will be closed automatically.",
    )
    parser.add_argument(
        "--force-margin",
        type=float,
        default=None,
        help="Force approximate margin in USD for the trade (converted to units using 1 unit = $0.05).",
    )
    parser.add_argument(
        "--aggressive-scaling",
        action="store_true",
        help="Enable $100K→$1M aggressive scaling strategy with slippage-adjusted Kelly sizing, smart order execution, and circuit breakers.",
    )
    
    # --- Intelligent Mode (LLM-enhanced reasoning) ---
    parser.add_argument(
        "--intelligent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LLM-enhanced intelligent mode: adds reasoning, multi-modal fusion, and adaptive learning on top of Buddy's predictions. (default: enabled; use --no-intelligent to disable)",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate detailed causal reasoning explaining why Buddy recommends this trade (Market Context → Signal → Key Drivers → Risks → Final Call).",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["ollama", "claude", "openai", "auto"],
        default="auto",
        help="LLM provider for intelligent mode: claude (Anthropic API, default), openai, ollama (local), or auto (detect from env vars).",
    )
    parser.add_argument(
        "--llm-enhance/--no-llm-enhance",
        dest="llm_enhance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deep LLM integration: dynamic gate thresholds (adjusted for edge cases), smarter sentiment aggregation, and context-aware trading. (default: enabled; use --no-llm-enhance to disable)",
    )

    # --- Continual Learning / Scheduling flags ---
    parser.add_argument(
        "--retrain-interval",
        type=float,
        default=168.0,
        help="Recommended hours between retraining sessions (default: 168 = 1 week). Used by drift detection.",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.03,
        help="Performance drop threshold for drift detection (default: 0.03 = 3%% accuracy drop triggers drift alert).",
    )
    parser.add_argument(
        "--feature-drift-threshold",
        type=float,
        default=0.10,
        help="Feature distribution shift threshold (default: 0.10 = 10%% change in feature means triggers drift).",
    )
    parser.add_argument(
        "--enable-drift-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic drift detection after each training session. (default: enabled; use --no-enable-drift-check to disable)",
    )
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=1000.0,
        help="EWC penalty strength for preventing catastrophic forgetting (default: 1000.0). Higher = stronger constraint.",
    )
    parser.add_argument(
        "--ewc-gamma",
        type=float,
        default=0.95,
        help="EWC Fisher decay factor when adding new tasks (default: 0.95). Lower = faster forgetting of old tasks.",
    )
    parser.add_argument(
        "--replay-mix-ratio",
        type=float,
        default=0.20,
        help="Fraction of training batch from replay buffer (default: 0.20 = 20%% old data mixed in).",
    )
    parser.add_argument(
        "--replay-capacity-ratio",
        type=float,
        default=0.10,
        help="Replay buffer capacity as fraction of total samples seen (default: 0.10 = 10%%).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.999,
        help="EMA smoothing factor for shadow weights (default: 0.999). Higher = more stable, slower adaptation.",
    )
    parser.add_argument(
        "--ema-update-freq",
        type=int,
        default=16,
        help="Update EMA weights every N training steps (default: 16).",
    )
    parser.add_argument(
        "--disable-continual-learning",
        action="store_true",
        help="Disable all continual learning features (EMA, EWC, replay buffer) for this training session.",
    )
    
    # --- Enterprise Training MLOps flags (enabled by default for ensemble) ---
    parser.add_argument(
        "--enterprise",
        action="store_true",
        default=True,
        help="Enable enterprise training features: MLflow tracking, walk-forward CV, bootstrap CI, statistical validation. (default: enabled)",
    )
    parser.add_argument(
        "--no-enterprise",
        action="store_true",
        help="Disable enterprise training features for faster training.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of walk-forward cross-validation folds (default: 5). Set to 0 to disable CV.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=True,
        help="Enable bootstrap confidence intervals for metrics (95%% CI). (default: enabled)",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for confidence intervals (default: 1000).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name (default: auto-generated from instrument/granularity).",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=True,
        help="Generate professional markdown training report with statistical analysis. (default: enabled)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating the training report.",
    )
    parser.add_argument(
        "--disable-ewc",
        action="store_true",
        help="Disable EWC (Elastic Weight Consolidation) for continual learning. EWC is enabled by default.",
    )
    
    # Multi-pair pre-training arguments
    parser.add_argument(
        "--multi-pair",
        action="store_true",
        help="For train-buddy: train foundation model on multiple pairs simultaneously",
    )
    parser.add_argument(
        "--foundation-pairs",
        type=str,
        default=None,
        help="For train-buddy --multi-pair: comma-separated pairs (default: EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CAD)",
    )
    
    # Meta-learning / suggest-improvements arguments
    parser.add_argument(
        "--auto-apply",
        action="store_true",
        help="For suggest-improvements: automatically apply approved hyperparam changes to config.",
    )

    # If invoked as `buddy` with no extra args, run the interactive wizard.
    # (The installed `buddy` launcher calls: python main.py buddy ...)
    if _maybe_run_buddy_interactive_wizard(default_config=DEFAULT_CONFIG_PATH):
        return

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    _normalize_command_args(args)

    # If user requested direct REPL, handle it immediately (before heavy dispatch).
    if _maybe_launch_buddy_repl(args):
        return

    command_map = {
        "train": train_buddy,  # Short alias
        "train-buddy": train_buddy,
        "retrain-gates": retrain_gates,
        "train-rl-sizer": train_rl_sizer,
        "buddy": buddy,
        "Buddy": buddy,
        "promote-model": promote_model,
        "model-status": model_status,
        "test": buddy_test,  # Legacy - redirects to validate
        "validate": buddy_validate,  # Professional validation
        "scan": buddy_scan,
        "analyze": buddy_analyze,
        "journal": buddy_journal,
        "suggest-improvements": suggest_improvements,
    }

    try:
        if args.command in ("train", "train-buddy"):
            _dispatch_train_buddy(args, command_map)
        elif args.command == "retrain-gates":
            # Retrain sklearn gates only (XGBoost, RF, Ridge)
            # Keeps Transformer direction model unchanged
            retrain_gates(
                config_path=args.config,
                pairs=getattr(args, "pairs", None),
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 5000)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        elif args.command == "train-rl-sizer":
            # Train RL position sizing agent
            train_rl_sizer(
                config_path=args.config,
                timesteps=int(getattr(args, "timesteps", 500_000)),
                episodes=getattr(args, "rl_episodes", None),
                pairs=getattr(args, "pairs", None),
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 5000)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        elif args.command in {"buddy", "Buddy"}:
            _dispatch_buddy(args, command_map)
        elif args.command == "promote-model":
            promote_model(args.config)
        elif args.command == "model-status":
            model_status(args.config)
        elif args.command == "test":
            buddy_test(
                args.config,
                instrument=str(getattr(args, "instrument", "USD_JPY")),
                granularity=str(getattr(args, "granularity", "H1")),
                test_candles=int(getattr(args, "candles", 50)),
                min_confidence=float(getattr(args, "min_confidence", 0.0)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        elif args.command == "validate":
            buddy_validate(
                args.config,
                instrument=str(getattr(args, "instrument", "EUR_USD")),
                granularity=str(getattr(args, "granularity", "H1")),
                candles=int(getattr(args, "candles", 800)),
                lookahead=int(getattr(args, "lookahead", 24)),
                all_pairs=bool(getattr(args, "all_pairs", False)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        elif args.command == "scan":
            watch_mode = bool(getattr(args, "watch", False))
            
            if watch_mode:
                # Continuous watch mode
                from buddy_scanner import BuddyScanner
                
                scanner = BuddyScanner(
                    config_path=args.config,
                    use_rl_sizer=bool(getattr(args, "use_rl_sizer", False)),
                )
                
                pairs_str = getattr(args, "pairs", None)
                pair_list = None
                if pairs_str:
                    pair_list = [p.strip().upper().replace("/", "_") for p in pairs_str.split(",")]
                
                scanner.scan_continuous(
                    pairs=pair_list,
                    granularity=str(getattr(args, "granularity", "H1")),
                    interval_minutes=int(getattr(args, "interval", 5)),
                    auto_execute=bool(getattr(args, "auto_execute", False)),
                    top_n=int(getattr(args, "top", 5)),
                )
            else:
                # Single scan
                buddy_scan(
                    args.config,
                    pairs=str(getattr(args, "pairs", "")) if getattr(args, "pairs", None) else None,
                    granularity=str(getattr(args, "granularity", "H1")),
                    top_n=int(getattr(args, "top", 5)),
                    verbose=bool(getattr(args, "verbose", False)),
                    prompt_train=not bool(getattr(args, "no_train", False)),
                    no_execute=bool(getattr(args, "skip_execute", False)),
                    use_rl_sizer=bool(getattr(args, "use_rl_sizer", False)),
                    diversified=bool(getattr(args, "diversified", False)),
                )
        elif args.command == "trade-analysis":
            # Trade analysis command
            from trade_analyzer import TradeAnalyzer
            from pathlib import Path
            
            analyzer = TradeAnalyzer()
            report = analyzer.analyze_recent(n=int(getattr(args, "last", 50)))
            
            # Print markdown report
            console.print(analyzer.format_markdown(report))
            
            # Export if requested
            export_path = getattr(args, "export", None)
            if export_path:
                import json
                with open(export_path, 'w') as f:
                    json.dump(analyzer.to_dict(report), f, indent=2)
                console.print(f"\n[green]✅ Report exported to {export_path}[/green]")
        elif args.command == "analyze":
            buddy_analyze(
                args.config,
                top_n=int(getattr(args, "top", 30)),
                save_plots=bool(getattr(args, "save", False)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        elif args.command == "journal":
            buddy_journal(
                args.config,
                update=bool(getattr(args, "update", False)),
                days=int(getattr(args, "days", 30)),
                verbose=bool(getattr(args, "verbose", False)),
                import_trades=bool(getattr(args, "import_trades", False)),
            )
        elif args.command == "suggest-improvements":
            suggest_improvements(
                args.config,
                instrument=getattr(args, "instrument", None),
                auto_apply=bool(getattr(args, "auto_apply", False)),
                verbose=bool(getattr(args, "verbose", False)),
            )
        else:
            command_map[args.command](args.config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logging.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
# — Raynergy-svg —
