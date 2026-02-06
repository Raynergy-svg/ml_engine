#!/usr/bin/env python3
"""I/O utilities for ML Engine Trading Bot CLI."""
from __future__ import annotations

import os
import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel

from src.utils import setup_logging, load_config


# Constants
DEFAULT_CONFIG_PATH = "./config/config_improved_H1.yaml"
DEFAULT_MESSAGE_FORMAT = "Epoch {epoch} completed"
TIMESTAMP_FORMAT = "%H:%M:%S"
TABLE_HEADER_STYLE = "bold magenta"
DEFAULT_CURRICULUM_KS = "32,64,128,0"
UTC_OFFSET_SUFFIX = "+00:00"
BUDDY_META_FILENAME = "buddy_tf.meta.json"

# Initialize console and logging
console = Console()
logger = setup_logging(log_file="cli.log")


@dataclass(frozen=True)
class OandaFetchOptions:
    instrument: str = "USD_JPY"
    granularity: str = "M5"
    candles: int = 5000
    price: str = "MBA"
    save_csv: str | None = None


# Valid OANDA instruments
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


def _oanda_fetch_to_csv(opts: OandaFetchOptions) -> str:
    from datetime import datetime, timezone

    import pandas as pd
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from fx_paper import candles_to_ohlcv_df
    from src.utils.oanda_practice import OandaPracticeClient

    client = OandaPracticeClient.from_env()

    # OANDA v20 candles endpoint typically caps `count` at 5000.
    # If the user requests more, page forward using `from_time`.
    max_per_request = 5000
    total = max(1, int(opts.candles))
    
    # Show progress spinner while fetching
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"Downloading {total:,} candles from OANDA ({opts.instrument} {opts.granularity})...",
            total=None,
        )
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
            for i in range(max_pages):
                if remaining <= 0:
                    break
                count = int(min(max_per_request, remaining))
                progress.update(task, description=f"Downloading {opts.instrument} {opts.granularity}: page {i+1}/{max_pages} ({total - remaining:,}/{total:,} candles)...")
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
        f"[bold]Market Data Retrieved Successfully[/bold]\n\n"
        f"[dim]Timestamp:[/dim] {now_utc_str}\n"
        f"[dim]Temporal Range:[/dim] {candle_range_str}\n"
        f"[dim]Observations:[/dim] {len(oanda_df):,} candles\n"
        f"[dim]Storage:[/dim] {out_path}",
        title="✓ OANDA API",
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

    console.print("[yellow]Initiating Keras 2.x → 3.x model migration...[/yellow]")

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
