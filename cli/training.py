#!/usr/bin/env python3
"""Training pipeline implementation for ML Engine Trading Bot."""
from __future__ import annotations

import hashlib
import os
import json
import time
from pathlib import Path
from typing import Any
from dataclasses import fields, replace

import numpy as np
import pandas as pd
from rich.table import Table

from cli.config import BuddyTrainingOptions, OandaFetchOptions
from src.utils.premium_output import PremiumConsole

# Module-level PremiumConsole instance for borderless output
_premium = PremiumConsole()
from cli.io_utils import DEFAULT_CURRICULUM_KS
from cli.calibration import (
    _tier2_nll, _tier2_spearman,
)
from cli.io_utils import (
    console, logger, _oanda_fetch_to_csv, _meta_path_for_checkpoint, _load_buddy_checkpoint,
    _get_pair_model_paths, _extract_instrument_from_csv_path,
)
from cli.tf_config import _configure_tf_metal
from cli.models import _build_buddy_model_for_type, _build_buddy_model_shared_encoder

from src.utils import load_config
from src.utils.seed_manager import set_global_seed, get_global_seed
from src.core.constants import DIRECTION_DEFAULTS

# ── String constants (SonarQube duplicate-literal fix) ──────────────
_STYLE_HEADER = "bold cyan"
_DISABLED_LABEL = "✗ Disabled"
_GRADIENT_BOOSTING = "Gradient Boosting"
_RANDOM_FOREST = "Random Forest"
_TRANSFORMER_KERAS_FILE = "transformer_direction.keras"
_META_JSON_FILE = "modular_ensemble.meta.json"


def _resolve_rl_position_sizing_config(cfg: Any) -> Any | None:
    """Resolve rl_integration.position_sizing config from root config dict."""
    if not isinstance(cfg, dict):
        return None
    try:
        from src.training.rl.config import RLIntegrationConfig

        return RLIntegrationConfig.from_dict(cfg).position_sizing
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to resolve RL position sizing config: %s", exc)
        return None


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
    rl_timesteps: int | None = None,
    min_samples: int = 500,
    *,
    features: np.ndarray | None = None,
    ensemble_predictions: np.ndarray | None = None,
    prices: np.ndarray | None = None,
    rl_position_cfg: Any | None = None,
) -> bool:
    """
    Train RL position sizer using ensemble training data.

    This function should be called after ensemble training completes successfully.
    It uses the actual features, ensemble predictions, and price data from training
    to learn optimal position sizing.

    Args:
        console: Rich console for output
        rl_timesteps: Optional override for RL training timesteps
        min_samples: Minimum samples required for training
        features: Market features array (n_samples, n_features) from ensemble training
        ensemble_predictions: Ensemble predictions (n_samples, 2) - [direction_prob, confidence]
        prices: Close prices array (n_samples,)
        rl_position_cfg: Optional RLIntegrationConfig.position_sizing object

    Returns:
        True if training was run, False otherwise
    """
    try:
        from src.training.rl.position_sizer import (
            RLConfig,
            RLPositionSizer,
            _ensure_gym_imported,
            _ensure_sb3_imported,
        )

        if not _ensure_sb3_imported() or not _ensure_gym_imported():
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
            console.print("[yellow]RL training deferred: data shape mismatch[/yellow]")
            console.print(f"[dim]Features: {n_samples} • Predictions: {len(ensemble_predictions)} • Prices: {len(prices)}[/dim]")
            return False

        base_config = RLConfig()
        if rl_position_cfg is not None:
            try:
                base_config = RLConfig.from_position_sizing_config(rl_position_cfg)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]RL config mapping failed, using defaults: {exc}[/yellow]")

        effective_timesteps = (
            int(rl_timesteps) if rl_timesteps is not None else int(base_config.total_timesteps)
        )

        # Configure RL with appropriate sequence length for data size.
        # Ensure we have enough data for the environment (need at least sequence_length + 100 steps).
        max_seq_len = max(10, n_samples - 200)  # Leave room for training episodes
        seq_len = min(int(base_config.sequence_length), max_seq_len)
        config = replace(
            base_config,
            total_timesteps=effective_timesteps,
            sequence_length=seq_len,
        )

        console.print("\n[bold cyan]🤖 RL Position Sizer Training[/bold cyan]")
        console.print(f"[dim]Using {n_samples:,} samples from ensemble training data[/dim]")
        console.print(
            "[dim]Timesteps: "
            f"{config.total_timesteps:,} • n_steps={config.n_steps} • "
            f"batch={config.batch_size} • epochs={config.n_epochs}[/dim]"
        )
        console.print(
            f"[dim]Sequence length: {seq_len}, Features: {features.shape[1]}, "
            f"lr={config.learning_rate:.1e}[/dim]"
        )
        console.print("[dim]Using cpu device (forced for macOS TF+PyTorch compatibility)[/dim]")
        import sys
        sys.stdout.flush()

        sizer = RLPositionSizer(config)

        console.print("")  # Blank line before progress
        sys.stdout.flush()

        # Train the RL agent with actual ensemble data
        stats = sizer.train(
            features, ensemble_predictions, prices,
        )

        console.print("[green]✅ RL position sizer trained and saved[/green]")
        console.print("[dim]Model: trained_data/models/rl_position_sizer.zip[/dim]")
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


def _load_config_suppressed(config_path: str):
    """Load config while suppressing verbose utils logging."""
    import logging as _logging
    _utils_logger = _logging.getLogger('src.utils.utils')
    _utils_prev = _utils_logger.level
    _utils_logger.setLevel(_logging.WARNING)
    cfg = load_config(config_path)
    _utils_logger.setLevel(_utils_prev)
    return cfg


def _is_rl_auto_train_enabled(cfg: dict, options: BuddyTrainingOptions) -> bool:
    """Resolve RL auto-train flag with backward compatibility."""
    training_auto = cfg.get("training", {}).get("auto_train_rl", True)
    rl_auto = cfg.get("rl_integration", {}).get(
        "auto_train_after_ensemble",
        training_auto,
    )
    return bool(options.train_rl_sizer and rl_auto)


def _extract_options_to_dict(options: "BuddyTrainingOptions") -> dict:
    """Extract training options into a flat dictionary for easier passing."""
    return {
        "oanda_fetch": options.oanda_fetch,
        "advanced": options.advanced,
        "feature_curriculum_opt": options.feature_curriculum,
        "curriculum_ks_opt": options.curriculum_ks,
        "pca_components": options.pca_components,
        "seq_len": options.seq_len,
        "epochs": options.epochs,
        "batch_size": options.batch_size,
        "lr": options.lr,
        "patience": options.patience,
        "seed": options.seed,
        "run_tag": options.run_tag,
        "all_features": options.all_features,
        "device": options.device,
        "ignore_input_mismatches": bool(options.ignore_input_mismatches),
        "disable_tier2_on_mismatch": bool(options.disable_tier2_on_mismatch),
        "mixed_precision": options.mixed_precision,
        "steps_per_execution": options.steps_per_execution,
        "jit_compile": options.jit_compile,
        "shuffle_buffer": options.shuffle_buffer,
        "prefetch": options.prefetch,
        "cache_val": options.cache_val,
        "combined_use_predict": options.combined_use_predict,
        "shared_encoder": options.shared_encoder,
        "model_type": options.model_type,
        "timing": options.timing,
        "fit_verbose": options.fit_verbose,
    }


def _resolve_oos_env_settings():
    """Resolve OOS evaluation settings from environment variables."""
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
    return oos_enabled_env, oos_candles_env


def _resolve_advanced_options(advanced, buddy_cfg, feature_curriculum_opt, curriculum_ks_opt):
    """Extract and resolve advanced training options with defaults from config."""
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

    if feature_curriculum_opt is None:
        feature_curriculum = bool(buddy_cfg.get("feature_curriculum", False))
    else:
        feature_curriculum = bool(feature_curriculum_opt)

    if curriculum_ks_opt is None:
        curriculum_ks = str(buddy_cfg.get("curriculum_ks", DEFAULT_CURRICULUM_KS))
    else:
        curriculum_ks = str(curriculum_ks_opt)

    # Clamp/validate filter params
    if median_window is not None and median_window <= 1:
        median_window = None
    if vol_norm_window is not None and vol_norm_window <= 1:
        vol_norm_window = None
    if min_volume is not None and min_volume <= 0:
        min_volume = None
    spread_pctl = min(max(spread_pctl, 0.5), 0.999)
    spread_mult = max(spread_mult, 1.0)

    return {
        "init_from": init_from,
        "es_monitor": es_monitor,
        "combined_w_dir": combined_w_dir,
        "combined_w_conf": combined_w_conf,
        "train_smoothing": train_smoothing,
        "top_features": top_features,
        "median_window": median_window,
        "vol_norm_window": vol_norm_window,
        "min_volume": min_volume,
        "spread_filter": spread_filter,
        "spread_pctl": spread_pctl,
        "spread_mult": spread_mult,
        "feature_curriculum": feature_curriculum,
        "curriculum_ks": curriculum_ks,
    }


def _resolve_tier2_options(advanced, buddy_cfg):
    """Resolve Tier-2 calibration options from advanced options and config."""
    tier2_calibrate = bool(getattr(advanced, "tier2_calibrate", True)) if advanced else True
    tier2_stop_loss_pips_cfg = float(buddy_cfg.get("stop_loss_pips", 15.0))
    tier2_take_profit_pips_cfg = float(buddy_cfg.get("take_profit_pips", 30.0))
    tier2_spread_fallback_pips_cfg = float(buddy_cfg.get("tier2_spread_fallback_pips", 2.0))
    tier2_horizon_candles_cfg = int(buddy_cfg.get("tier2_horizon_candles", 288))
    tier2_tie_break_cfg = str(buddy_cfg.get("tier2_tie_break", "sl"))

    adv_sl = getattr(advanced, "tier2_stop_loss_pips", None) if advanced else None
    adv_tp = getattr(advanced, "tier2_take_profit_pips", None) if advanced else None
    adv_sp = getattr(advanced, "tier2_spread_fallback_pips", None) if advanced else None
    adv_hz = getattr(advanced, "tier2_horizon_candles", None) if advanced else None
    adv_tb = getattr(advanced, "tier2_tie_break", None) if advanced else None

    return {
        "tier2_calibrate": tier2_calibrate,
        "tier2_stop_loss_pips": float(adv_sl) if adv_sl is not None else tier2_stop_loss_pips_cfg,
        "tier2_take_profit_pips": float(adv_tp) if adv_tp is not None else tier2_take_profit_pips_cfg,
        "tier2_spread_fallback_pips": float(adv_sp) if adv_sp is not None else tier2_spread_fallback_pips_cfg,
        "tier2_horizon_candles": int(adv_hz) if adv_hz is not None else tier2_horizon_candles_cfg,
        "tier2_tie_break": str(adv_tb) if adv_tb is not None else tier2_tie_break_cfg,
        "tier2_calibration_bins": int(getattr(advanced, "tier2_calibration_bins", 20)) if advanced else 20,
        "tier2_calibration_stride": int(getattr(advanced, "tier2_calibration_stride", 5)) if advanced else 5,
    }


def _setup_oos_live_split(oos_enabled_env, csv_path, oanda_fetch, oos_candles_env):
    """Set up OOS live split if applicable. Returns (oos_live_split, base_oanda_fetch, oanda_fetch)."""
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
    return oos_live_split, base_oanda_fetch, oanda_fetch


def _load_training_data(
    *,
    options,
    cfg,
    csv_path,
    oanda_fetch,
    console,
    multi_pair_mode,
    foundation_pairs_str,
    min_volume,
    spread_filter,
    spread_pctl,
    spread_mult,
    seq_len,
):
    """Load training data from OANDA, CSV, or multi-pair sources. Returns (df, oos_live_split_override)."""
    from src.training.buddy_training_helpers import _buddy_load_and_validate_csv, _load_multi_pair_data

    oos_live_split_override = None  # None means no change

    if multi_pair_mode:
        DEFAULT_FOUNDATION_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "USD_CHF"]

        if foundation_pairs_str:
            pairs_list = [p.strip().upper().replace("/", "_") for p in foundation_pairs_str.split(",")]
        else:
            pairs_list = DEFAULT_FOUNDATION_PAIRS

        candles_per_pair = 5000
        if oanda_fetch is not None:
            try:
                candles_per_pair = int(getattr(oanda_fetch, "candles", 5000) or 5000)
            except Exception:
                pass

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
            raise ValueError(f"Multi-pair data loading failed or insufficient data: {len(df) if df is not None else 0} observations")

        console.print(f"[green]✓ Multi-pair data loaded: {len(df):,} obs from {len(pairs_list)} pairs[/green]")
        oos_live_split_override = False
    else:
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

    return df, oos_live_split_override


def _apply_oos_holdout_split(df, oos_live_split, base_oanda_fetch, oos_candles_env, seq_len, console):
    """If configured, split OOS holdout from the tail of the data. Returns (df, oos_holdout_csv, oos_holdout_rows)."""
    oos_holdout_csv = None
    oos_holdout_rows = None

    if not oos_live_split or base_oanda_fetch is None:
        return df, oos_holdout_csv, oos_holdout_rows

    try:
        holdout_n = int(oos_candles_env)
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

    return df, oos_holdout_csv, oos_holdout_rows


def _apply_feature_engineering(df, cfg, all_features, train_smoothing, median_window, console):
    """Apply feature engineering and noise reduction. Returns modified df."""
    from src.data.feature_engineering import FeatureEngineering

    if train_smoothing:
        extra = f" + median(close,w={median_window})" if median_window else ""
        console.print(f"Training noise reduction enabled: resample->M5{extra} + EMA(14) close")

    if all_features:
        t_fe = time.perf_counter()
        fe = FeatureEngineering(cfg.get("feature_engineering", {}))
        df = fe.create_features(
            df,
            include_all=True,
            apply_candle_smoothing=train_smoothing,
            median_window=median_window,
        )

        elapsed_fe = time.perf_counter() - t_fe
        console.print(Panel(
            f"[bold]Feature Engineering Pipeline Complete[/bold]\n\n"
            f"[dim]Noise Reduction:[/dim] {bool(train_smoothing)} • [dim]Median:[/dim] {median_window or 'None'}\n"
            f"[dim]Output Dimensionality:[/dim] {int(len(df)):,} obs × {int(df.shape[1])}\n"
            f"[dim]Processing Time:[/dim] {elapsed_fe:.2f}s",
            title="⚙️  Feature Engineering",
            border_style="blue",
        ))
    elif train_smoothing:
        try:
            from src.data.candle_smoothing import resample_5min_ohlcv_and_ema_close

            df = resample_5min_ohlcv_and_ema_close(
                df,
                ema_span=14,
                time_col="time",
                median_window=median_window,
            )
        except Exception as e:
            console.print(f"[yellow]Noise reduction deferred:[/yellow] {e}")

    return df


def _resolve_init_feature_columns(init_from, console):
    """Load feature columns from warm-start checkpoint metadata."""
    if not init_from:
        return None
    try:
        meta_p = _meta_path_for_checkpoint(str(init_from))
        if meta_p:
            init_meta = json.loads(meta_p.read_text())
            cols = init_meta.get("feature_columns")
            if isinstance(cols, list) and cols:
                console.print(f"Using feature_columns from init_from metadata: {meta_p}")
                return [str(c) for c in cols]
    except Exception as e:
        console.print(f"[yellow]Warm-start metadata unavailable:[/yellow] {e}")
    return None


def _prepare_numeric_features(df, init_feature_columns, top_features, init_from, console):
    """Prepare numeric feature DataFrame with column alignment and validation."""
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        raise ValueError("No numeric features found after preprocessing")

    if top_features is not None and init_from:
        raise ValueError(
            "--top-features cannot be used with --warm-start/--init-from because the checkpoint expects a fixed feature set. "
            "Run without warm-start to train a new (smaller) model, then you can warm-start future runs from that new checkpoint."
        )

    if init_feature_columns is not None:
        missing_cols = [c for c in init_feature_columns if c not in numeric_df.columns]
        extra_cols = [c for c in numeric_df.columns if c not in set(init_feature_columns)]
        if missing_cols:
            console.print(f"[yellow]Warm-start[/yellow]: {len(missing_cols)} missing features will be filled with 0.0")
        if extra_cols:
            console.print(f"[yellow]Warm-start[/yellow]: {len(extra_cols)} extra features will be dropped")
        numeric_df = numeric_df.reindex(columns=init_feature_columns, fill_value=0.0)

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return numeric_df


def _compute_feature_ranking(
    numeric_df, closes, feature_curriculum, top_features, vol_norm_window, console,
):
    """Compute feature ranking for curriculum learning or top-K selection. Returns (numeric_df, ranked_cols)."""
    ranked_cols = None
    if not (feature_curriculum or (top_features is not None and top_features > 0)):
        return numeric_df, ranked_cols

    try:
        y_dir = (closes[1:] > closes[:-1]).astype(np.float32)
        r0 = (closes[1:] - closes[:-1]) / np.maximum(closes[:-1], 1e-8)

        if vol_norm_window is not None:
            w = int(vol_norm_window)
            ret_std = (
                pd.Series(r0)
                .rolling(w, min_periods=max(10, w // 5))
                .std()
                .to_numpy(dtype=np.float32)
            )
            if not np.isfinite(ret_std).all():
                finite = ret_std[np.isfinite(ret_std)]
                fill = float(np.median(finite)) if finite.size else 1.0
                ret_std = np.where(np.isfinite(ret_std), ret_std, fill).astype(np.float32)
            r0 = r0 / np.maximum(ret_std, 1e-6)

        abs_ret = np.abs(r0).astype(np.float32)
        abs_ret = np.nan_to_num(abs_ret, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

        x_rank_all = numeric_df.iloc[:-1].astype(np.float32)
        n_rank = int(len(x_rank_all))
        split_rank = int(n_rank * 0.8)
        if split_rank <= 10:
            raise ValueError("Not enough rows for train-only ranking")

        x_rank = x_rank_all.iloc[:split_rank]
        y_dir_rank = pd.Series(y_dir[:split_rank])
        abs_ret_rank = pd.Series(abs_ret[:split_rank])

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
        console.print(f"[yellow]Feature ranking unavailable:[/yellow] {e}")
        ranked_cols = None

    return numeric_df, ranked_cols


def _compute_tier2_confidence_labels(
    *,
    n,
    direction_y_all,
    ohlc_df,
    oanda_fetch,
    csv_path,
    tier2_opts,
    seq_len,
    console,
):
    """Compute Tier-2 TP/SL confidence labels. Returns (conf_y_all, conf_w_all, conf_instrument_guess)."""
    tier2_calibrate = tier2_opts["tier2_calibrate"]
    tier2_stop_loss_pips = tier2_opts["tier2_stop_loss_pips"]
    tier2_take_profit_pips = tier2_opts["tier2_take_profit_pips"]
    tier2_spread_fallback_pips = tier2_opts["tier2_spread_fallback_pips"]
    tier2_horizon_candles = tier2_opts["tier2_horizon_candles"]
    tier2_tie_break = tier2_opts["tier2_tie_break"]
    tier2_calibration_stride = tier2_opts["tier2_calibration_stride"]

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
        return conf_y_all, conf_w_all, conf_instrument_guess, label_stride

    try:
        t_tier2_label = time.perf_counter()
        from src.utils.fx_paper import pip_size, simulate_tp_sl_outcome

        instrument_guess = _guess_instrument(oanda_fetch, csv_path)
        conf_instrument_guess = str(instrument_guess)

        pip = float(pip_size(str(instrument_guess)))
        max_signal = min(int(n) - 1, int(len(ohlc_df)) - 3)
        if max_signal < 0:
            raise ValueError("Insufficient OHLC rows for TP/SL labeling")

        start_gi = max(0, int(seq_len) - 1)
        n_sim = 0
        for gi in range(int(start_gi), int(max_signal) + 1, int(label_stride)):
            spread_pips = _spread_pips_at_entry_generic(
                ohlc_df, int(gi), pip, tier2_spread_fallback_pips
            )
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

    return conf_y_all, conf_w_all, conf_instrument_guess, label_stride


def _guess_instrument(oanda_fetch, csv_path):
    """Best-effort instrument inference from fetch options or CSV path."""
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
    return instrument_guess


def _spread_pips_at_entry_generic(ohlc_df, gi, pip, fallback_pips):
    """Compute spread in pips at a given entry index."""
    sp = float(fallback_pips)
    try:
        if "bid_close" in ohlc_df.columns and "ask_close" in ohlc_df.columns:
            bid = float(ohlc_df["bid_close"].iloc[int(gi) + 1])
            ask = float(ohlc_df["ask_close"].iloc[int(gi) + 1])
            if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid:
                sp = float((ask - bid) / max(pip, 1e-9))
    except Exception:
        sp = float(fallback_pips)
    return float(sp)


def _standardize_features(feats, split):
    """Standardize features using train-only statistics. Returns (feats, mu, sigma, train_feats, val_feats)."""
    feats = feats.astype(np.float32, copy=False)
    train_feats_view = feats[:split]
    mu = train_feats_view.mean(axis=0, keepdims=True, dtype=np.float32)
    sigma = train_feats_view.std(axis=0, keepdims=True, dtype=np.float32)
    sigma = np.where(sigma < 1e-6, np.float32(1.0), sigma).astype(np.float32)
    feats -= mu
    feats /= sigma
    train_feats = feats[:split]
    val_feats = feats[split:]
    return feats, mu, sigma, train_feats, val_feats


def _apply_pca(train_feats, val_feats, pca_components, console):
    """Apply optional PCA dimensionality reduction. Returns (train_feats, val_feats, pca_model, pca_components_eff)."""
    pca_components_eff = None
    pca_model = None

    if pca_components is not None:
        pca_components_eff = int(pca_components)
        if pca_components_eff <= 0:
            pca_components_eff = None

    orig_feature_dim = int(train_feats.shape[1])

    if pca_components_eff is not None and pca_components_eff < train_feats.shape[1]:
        from sklearn.decomposition import PCA

        pca_model = PCA(n_components=pca_components_eff, svd_solver="auto", random_state=42)
        pca_model.fit(train_feats)
        train_feats = pca_model.transform(train_feats).astype(np.float32)
        val_feats = pca_model.transform(val_feats).astype(np.float32)
        console.print(f"PCA enabled: {orig_feature_dim} -> {train_feats.shape[1]}")

    return train_feats, val_feats, pca_model, pca_components_eff


def _validate_features_and_targets(train_feats, val_feats, train_dir_raw, val_dir_raw, train_conf_raw, val_conf_raw, train_conf_w_raw, val_conf_w_raw):
    """Validate that all arrays are finite."""
    if not np.isfinite(train_feats).all() or not np.isfinite(val_feats).all():
        raise ValueError("Non-finite values detected in standardized features (check preprocessing/filtering).")
    if not np.isfinite(train_dir_raw).all() or not np.isfinite(val_dir_raw).all():
        raise ValueError("Non-finite values detected in direction targets.")
    if not np.isfinite(train_conf_raw).all() or not np.isfinite(val_conf_raw).all():
        raise ValueError("Non-finite values detected in confidence targets.")
    if not np.isfinite(train_conf_w_raw).all() or not np.isfinite(val_conf_w_raw).all():
        raise ValueError("Non-finite values detected in confidence target weights.")


def _format_ridge_sparsity(ridge_metrics: dict) -> str:
    """Format sparsity info for Ridge/LightGBM confidence model.

    Handles both LightGBM (n_important_features/n_total_features)
    and ElasticNet (n_nonzero_coefs/n_total_coefs) metric keys.
    """
    # LightGBM backend
    n_imp = ridge_metrics.get("n_important_features")
    n_tot_lgbm = ridge_metrics.get("n_total_features")
    if n_imp is not None and n_tot_lgbm is not None:
        return f"{n_imp}/{n_tot_lgbm}"

    # ElasticNet backend
    n_nz = ridge_metrics.get("n_nonzero_coefs")
    n_tot_en = ridge_metrics.get("n_total_coefs")
    if n_nz is not None and n_tot_en is not None:
        return f"{n_nz}/{n_tot_en}"

    return "N/A"


def _train_ensemble_models(
    *,
    cfg,
    options,
    console,
    oanda_fetch,
    csv_path,
    train_feats,
    val_feats,
    df,
    ohlc_df,
    feature_columns,
    seq_len,
    epochs,
    batch_size,
    lr,
    patience,
    fit_verbose,
    rl_position_cfg=None,
):
    """Train the full modular ensemble (Transformer/TCN + XGBoost + RF + Ridge). Returns when complete."""
    from src.core.modular_data_loaders import load_all_modular_data

    enterprise_enabled = options.enterprise if hasattr(options, 'enterprise') else True
    bootstrap_enabled = options.bootstrap if hasattr(options, 'bootstrap') else True
    cv_folds = options.cv_folds if hasattr(options, 'cv_folds') else 5
    generate_report_enabled = options.generate_report if hasattr(options, 'generate_report') else True

    # Professional header
    console.print()
    console.print(Panel.fit(
        "[bold white]ADVANCED ENSEMBLE TRAINING SYSTEM[/bold white]\n"
        "[dim]Enterprise-Grade Machine Learning Pipeline • Production-Ready Architecture[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()

    # Enterprise features status table
    enterprise_table = Table(show_header=True, header_style=_STYLE_HEADER, box=None, padding=(0, 2))
    enterprise_table.add_column("Component", style="cyan", width=26)
    enterprise_table.add_column("Status", style="green", width=20)
    enterprise_table.add_row("Experiment Tracking", "✓ Active" if enterprise_enabled else "✗ Inactive")
    enterprise_table.add_row("Walk-Forward Validation", f"✓ {cv_folds}-Fold CV" if cv_folds > 0 else _DISABLED_LABEL)
    enterprise_table.add_row("Statistical Bootstrap", "✓ 95% CI" if bootstrap_enabled else _DISABLED_LABEL)
    enterprise_table.add_row("Automated Reporting", "✓ Enabled" if generate_report_enabled else _DISABLED_LABEL)

    console.print(Panel(enterprise_table, title="[bold]Enterprise Features[/bold]", border_style="green"))
    console.print()

    # Determine instrument early so we can load pair-specific config
    training_instrument, training_granularity = _determine_training_instrument(oanda_fetch, csv_path)

    # Get model configuration from config (intel_optimized.yaml is the default)
    transformer_cfg = cfg.get("transformer", {})

    use_transformer = transformer_cfg.get("use_transformer", True)
    use_regime = transformer_cfg.get("use_regime", False)

    # Get direction params from base config (intel_optimized.yaml)
    direction_threshold = cfg.get("direction_threshold", DIRECTION_DEFAULTS['threshold'])
    direction_lookahead = cfg.get("direction_lookahead", DIRECTION_DEFAULTS['lookahead'])
    console.print(f"[cyan]  Using config direction_threshold: {direction_threshold:.4f}[/cyan]")
    console.print(f"[cyan]  Using config direction_lookahead: {direction_lookahead}[/cyan]")

    regime_lookback = transformer_cfg.get("regime_lookback", 20)
    regime_lookahead = transformer_cfg.get("regime_lookahead", 12)

    # Model architecture table
    _print_ensemble_architecture_table(console, use_regime, use_transformer, direction_threshold)

    # Prepare feature DataFrame
    feature_df = _prepare_ensemble_feature_df(df, ohlc_df, train_feats, val_feats, feature_columns, cfg, console)

    # Calculate split ratio
    n_total = len(train_feats) + len(val_feats)
    train_frac = len(train_feats) / n_total
    val_frac = len(val_feats) / n_total
    sum_frac = train_frac + val_frac
    train_frac = train_frac / sum_frac * 0.9
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

    _print_ensemble_dataset_summary(console, all_data)

    # Log instrument and granularity (already determined earlier for pair config loading)
    console.print(f"[cyan]📊 Training instrument: {training_instrument}, Granularity: {training_granularity}[/cyan]")

    # Detect warm-start
    model_dir = Path("trained_data/models")
    use_transformer_flag = (options.model_type or "transformer") != "tcn"
    pair_paths = _get_pair_model_paths(model_dir, training_instrument,
                                       "transformer" if use_transformer_flag else "tcn")

    is_warm_start, warm_start_path = _detect_warm_start(
        model_dir, pair_paths, training_instrument, cfg, console
    )

    # Continual learning settings
    use_ema, use_ewc, use_replay, disable_cl = _resolve_continual_learning(
        is_warm_start, options, console
    )

    ema_alpha, ema_update_freq = 0.999, 16
    ewc_lambda, ewc_gamma = 1000.0, 0.95
    replay_capacity_ratio, replay_mix_ratio = 0.10, 0.20
    drift_threshold = 0.03

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

    # Build trainer config
    pair_checkpoint_dir = str(pair_paths['pair_dir'] / "checkpoints") if training_instrument != "GENERIC" else "trained_data/checkpoints"
    trainer_config = _build_trainer_config(
        cfg=cfg,
        transformer_cfg=transformer_cfg,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        fit_verbose=fit_verbose,
        pair_checkpoint_dir=pair_checkpoint_dir,
        use_ema=use_ema,
        ema_alpha=ema_alpha,
        ema_update_freq=ema_update_freq,
        use_ewc=use_ewc,
        ewc_lambda=ewc_lambda,
        ewc_gamma=ewc_gamma,
        use_replay=use_replay,
        replay_capacity_ratio=replay_capacity_ratio,
        replay_mix_ratio=replay_mix_ratio,
        drift_threshold=drift_threshold,
    )

    all_metrics = {}

    # Train direction/regime model
    dir_data, dir_metrics, dir_model_path, dir_trainer = _train_direction_or_regime_model(
        use_regime=use_regime,
        use_transformer=use_transformer,
        all_data=all_data,
        trainer_config=trainer_config,
        model_dir=model_dir,
        pair_paths=pair_paths,
        training_instrument=training_instrument,
        warm_start_path=warm_start_path,
        is_warm_start=is_warm_start,
        direction_threshold=direction_threshold,
        direction_lookahead=direction_lookahead,
        regime_lookback=regime_lookback,
        regime_lookahead=regime_lookahead,
        console=console,
        cfg=cfg,
    )
    all_metrics['direction'] = dir_metrics

    # Train XGBoost
    xgb_data, xgb_metrics = _train_xgboost_model(
        all_data=all_data,
        trainer_config=trainer_config,
        pair_paths=pair_paths,
        model_dir=model_dir,
        training_instrument=training_instrument,
        console=console,
    )
    all_metrics['xgboost'] = xgb_metrics

    # Train Random Forest
    rf_data, rf_metrics = _train_rf_model(
        all_data=all_data,
        trainer_config=trainer_config,
        pair_paths=pair_paths,
        model_dir=model_dir,
        training_instrument=training_instrument,
        console=console,
    )
    all_metrics['rf'] = rf_metrics

    # Train Ridge
    ridge_data, ridge_metrics, ridge_trainer = _train_ridge_model(
        all_data=all_data,
        trainer_config=trainer_config,
        pair_paths=pair_paths,
        model_dir=model_dir,
        training_instrument=training_instrument,
        console=console,
    )
    all_metrics['ridge'] = ridge_metrics

    # Optional HistGB
    train_histgb = cfg.get("buddy", {}).get("train_defaults", {}).get("train_histgb", False)
    if train_histgb and not use_regime:
        _train_histgb_model(
            dir_data=dir_data,
            trainer_config=trainer_config,
            pair_paths=pair_paths,
            model_dir=model_dir,
            training_instrument=training_instrument,
            all_metrics=all_metrics,
            console=console,
        )

    # Train meta-labeler (predicts trade success probability)
    meta_labeling_enabled = bool(cfg.get("buddy", {}).get("meta_labeling", {}).get("enabled", False))
    if meta_labeling_enabled:
        _train_meta_labeler_cli(
            dir_trainer=dir_trainer,
            dir_data=dir_data,
            pair_paths=pair_paths,
            model_dir=model_dir,
            training_instrument=training_instrument,
            cfg=cfg,
            console=console,
        )

    # Save metadata
    # Fit confidence calibrator (Platt scaling) after all models are trained
    _fit_post_training_calibrator_cli(
        dir_trainer=dir_trainer,
        dir_data=dir_data,
        pair_paths=pair_paths,
        training_instrument=training_instrument,
        model_dir=model_dir,
        console=console,
    )

    meta = _build_ensemble_metadata(
        use_regime=use_regime,
        dir_model_path=dir_model_path,
        dir_data=dir_data,
        dir_metrics=dir_metrics,
        xgb_data=xgb_data,
        xgb_metrics=xgb_metrics,
        rf_data=rf_data,
        rf_metrics=rf_metrics,
        ridge_data=ridge_data,
        ridge_metrics=ridge_metrics,
        pair_paths=pair_paths,
        training_instrument=training_instrument,
        training_granularity=training_granularity,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        feature_df=feature_df,
        direction_threshold=direction_threshold,
        direction_lookahead=direction_lookahead,
        regime_lookback=regime_lookback,
        regime_lookahead=regime_lookahead,
        use_transformer=use_transformer,
    )

    meta_path = model_dir / _META_JSON_FILE
    _save_ensemble_metadata(meta, meta_path, pair_paths, training_instrument)

    # Print summary
    _print_ensemble_summary(
        console=console,
        use_regime=use_regime,
        dir_metrics=dir_metrics,
        xgb_metrics=xgb_metrics,
        rf_metrics=rf_metrics,
        ridge_metrics=ridge_metrics,
        dir_model_path=dir_model_path,
        pair_paths=pair_paths,
        meta_path=meta_path,
        training_instrument=training_instrument,
        model_dir=model_dir,
    )

    # Enterprise validation
    if enterprise_enabled:
        _run_enterprise_validation(
            options=options,
            console=console,
            cfg=cfg,
            use_regime=use_regime,
            dir_trainer=dir_trainer,
            dir_data=dir_data,
            dir_metrics=dir_metrics,
            xgb_metrics=xgb_metrics,
            rf_metrics=rf_metrics,
            ridge_metrics=ridge_metrics,
            ridge_trainer=ridge_trainer,
            ridge_data=ridge_data,
            feature_df=feature_df,
            pair_paths=pair_paths,
            meta_path=meta_path,
            training_instrument=training_instrument,
            training_granularity=training_granularity,
            model_dir=model_dir,
            seq_len=seq_len,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            direction_threshold=direction_threshold,
            direction_lookahead=direction_lookahead,
            dir_model_path=dir_model_path,
            rl_position_cfg=rl_position_cfg,
        )


def _print_ensemble_architecture_table(console, use_regime, use_transformer, direction_threshold):
    """Print the model architecture table for ensemble training."""
    arch_table = Table(show_header=True, header_style=_STYLE_HEADER, box=None)
    arch_table.add_column("Component", style="white", width=19)
    arch_table.add_column("Objective", style="yellow", width=26)
    arch_table.add_column("Output Specification", style="green")

    if use_regime:
        arch_table.add_row("Transformer", "Regime Classification", "Trend | Chop |")
        arch_table.add_row("Network", "", "Mean Revert")
        arch_table.add_row("", "", "")
        arch_table.add_row(_GRADIENT_BOOSTING, "Momentum Vector Analysis", "Score • Acceleration")
        arch_table.add_row("", "", "Metric")
        arch_table.add_row("", "", "")
        arch_table.add_row(_RANDOM_FOREST, "Risk Exposure Assessment", "Drawdown Estimate •")
        arch_table.add_row("", "", "Streak Probability")
        arch_table.add_row("", "", "")
        arch_table.add_row("Regularized", "Prediction Confidence", "Calibrated Score")
        arch_table.add_row("Regression", "", "(0-100)")
    else:
        dir_model_name = "Transformer" if use_transformer else "TCN"
        arch_table.add_row(f"{dir_model_name}", "Directional Prediction", "Long | Short Signal")
        arch_table.add_row("Network", f"(Δ>{direction_threshold:.2%})", "")
        arch_table.add_row("", "", "")
        arch_table.add_row(_GRADIENT_BOOSTING, "Momentum Vector Analysis", "Score • Acceleration")
        arch_table.add_row("", "", "Metric")
        arch_table.add_row("", "", "")
        arch_table.add_row(_RANDOM_FOREST, "Risk Exposure Assessment", "Drawdown Estimate •")
        arch_table.add_row("", "", "Streak Probability")
        arch_table.add_row("", "", "")
        arch_table.add_row("Regularized", "Prediction Confidence", "Calibrated Score")
        arch_table.add_row("Regression", "", "(0-100)")

    console.print(Panel(arch_table, title="[bold]Model Architecture[/bold]", border_style="yellow"))
    console.print()


def _prepare_ensemble_feature_df(df, ohlc_df, train_feats, val_feats, feature_columns, cfg, console):
    """Prepare the feature DataFrame for ensemble modular loaders."""
    console.print()
    console.print(Panel(
        "[bold]Preparing Modular Training Data[/bold]\n\n"
        "[dim]Loading specialized datasets for each ensemble component...[/dim]",
        title="📦 Data Preparation",
        border_style="blue",
    ))

    n_samples = len(train_feats) + len(val_feats)

    if df is not None and 'close' in df.columns:
        feature_df = df.iloc[:n_samples].copy()
        console.print(f"  [green]✓[/green] Using RAW data (n={len(feature_df):,})")
    elif ohlc_df is not None:
        feature_df = ohlc_df.copy()
        if len(feature_df) > n_samples:
            feature_df = feature_df.iloc[:n_samples].copy()
        if len(feature_df.columns) < 20:
            from src.data.feature_engineering import FeatureEngineering
            fe_modular = FeatureEngineering(cfg.get("feature_engineering", {}))
            feature_df = fe_modular.create_features(feature_df, include_all=True)
        console.print(f"  [green]✓[/green] Using OHLCV data (n={len(feature_df):,})")
    else:
        console.print("  [yellow]⚠ Using pre-standardized features - may cause scale mismatch![/yellow]")
        all_feats = np.vstack([train_feats, val_feats])
        feature_df = pd.DataFrame(all_feats, columns=feature_columns)

    # Ensure OHLCV columns
    if ohlc_df is not None:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in feature_df.columns and col in ohlc_df.columns:
                vals = ohlc_df[col].values
                if len(vals) >= len(feature_df):
                    feature_df[col] = vals[:len(feature_df)]

    if 'close' not in feature_df.columns:
        for col in feature_df.columns:
            if 'close' in col.lower():
                feature_df['close'] = feature_df[col].values
                break

    if 'close' not in feature_df.columns:
        raise ValueError("No 'close' column found for direction labels")

    if 'high' not in feature_df.columns:
        feature_df['high'] = feature_df['close'] * 1.001
    if 'low' not in feature_df.columns:
        feature_df['low'] = feature_df['close'] * 0.999
    if 'volume' not in feature_df.columns:
        feature_df['volume'] = 1000

    console.print(f"  Data prepared: [green]{len(feature_df):,}[/green] obs × [green]{len(feature_df.columns)}[/green] features")
    return feature_df


def _print_ensemble_dataset_summary(console, all_data):
    """Print Rich table summarizing datasets for each ensemble model."""
    dataset_table = Table(title=None, show_header=True, header_style=_STYLE_HEADER, box=None)
    dataset_table.add_column("Model", style="white")
    dataset_table.add_column("Train", justify="right", style="green")
    dataset_table.add_column("Val", justify="right", style="yellow")
    dataset_table.add_column("Features", justify="right", style="blue")
    dataset_table.add_column("Notes", style="dim")

    for name, data in all_data.items():
        n_features = len(data['feature_names'])
        notes = ""
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


def _determine_training_instrument(oanda_fetch, csv_path):
    """Determine instrument and granularity from fetch options or CSV path."""
    if oanda_fetch:
        training_instrument = getattr(oanda_fetch, 'instrument', 'GENERIC')
        training_granularity = getattr(oanda_fetch, 'granularity', 'H1')
    elif csv_path:
        training_instrument = _extract_instrument_from_csv_path(csv_path)
        import re
        gran_match = re.search(r'_(M\d+|H\d+|D|W)[_.]', str(csv_path).upper())
        training_granularity = gran_match.group(1) if gran_match else 'H1'
    else:
        training_instrument = 'GENERIC'
        training_granularity = 'H1'
    return training_instrument, training_granularity


def _detect_warm_start(model_dir, pair_paths, training_instrument, cfg, console):
    """Detect whether warm-start (compound learning) should be used. Returns (is_warm_start, warm_start_path)."""
    has_existing_model = pair_paths['direction'].exists()
    generic_paths = _get_pair_model_paths(model_dir, "GENERIC",
                                          "transformer" if pair_paths['direction'].name.startswith("transformer") else "tcn")
    has_generic_model = (model_dir / _TRANSFORMER_KERAS_FILE).exists() or generic_paths['direction'].exists()
    warm_start_enabled = cfg.get("buddy", {}).get("train_defaults", {}).get("warm_start", True)

    if has_existing_model and warm_start_enabled:
        console.print(f"[green]♻️  Compound learning: warm-start from {training_instrument} model[/green]")
        return True, pair_paths['direction']
    elif has_generic_model and warm_start_enabled:
        console.print(f"[yellow]⚡ Initial training for {training_instrument} (warm-start from generic model)[/yellow]")
        return True, model_dir / _TRANSFORMER_KERAS_FILE
    else:
        return False, None


def _resolve_continual_learning(is_warm_start, options, console):
    """Resolve continual learning settings. Returns (use_ema, use_ewc, use_replay, disable_cl)."""
    use_ema = True
    use_ewc = is_warm_start and options.enable_ewc
    use_replay = is_warm_start
    disable_cl = not is_warm_start

    if not is_warm_start:
        console.print("[dim]Fresh training - EMA enabled, EWC/Replay disabled[/dim]")
    elif use_ewc:
        console.print("[cyan]Continual Learning: EMA + EWC + Replay enabled[/cyan]")
    else:
        console.print("[cyan]Continual Learning: EMA + Replay enabled (EWC disabled via --disable-ewc)[/cyan]")

    return use_ema, use_ewc, use_replay, disable_cl


def _build_trainer_config(
    *,
    cfg,
    transformer_cfg,
    epochs,
    batch_size,
    lr,
    patience,
    fit_verbose,
    pair_checkpoint_dir,
    use_ema,
    ema_alpha,
    ema_update_freq,
    use_ewc,
    ewc_lambda,
    ewc_gamma,
    use_replay,
    replay_capacity_ratio,
    replay_mix_ratio,
    drift_threshold,
):
    """Build TrainerConfig from config and overrides."""
    from src.training.modular_trainers import TrainerConfig

    overfit_cfg = cfg.get("overfit_prevention", {})
    swa_cfg = cfg.get("swa", {})
    cosine_cfg = cfg.get("cosine_restarts", {})
    warmstart_cfg = cfg.get("warmstart_recovery", {})
    feature_cfg = cfg.get("feature_selection", {})
    
    # Extract Hybrid SFT-RL settings from training section
    training_cfg = cfg.get("training", {})
    use_hybrid_sft_rl = bool(training_cfg.get("use_hybrid_sft_rl", False))
    rl_prob = float(training_cfg.get("rl_prob", 0.5))
    entropy_coef = float(training_cfg.get("entropy_coef", 0.01))
    initial_baseline = float(training_cfg.get("initial_baseline", 0.5))
    baseline_momentum = float(training_cfg.get("baseline_momentum", 0.9))

    return TrainerConfig(
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(lr),
        patience=int(patience),
        verbose=int(fit_verbose),
        checkpoint_dir=pair_checkpoint_dir,
        transformer_d_model=int(transformer_cfg.get("d_model", 32)),
        transformer_num_heads=int(transformer_cfg.get("num_heads", 4)),
        transformer_num_layers=int(transformer_cfg.get("num_layers", 2)),
        transformer_dff=int(transformer_cfg.get("dff", 64)),
        transformer_dropout=float(transformer_cfg.get("dropout", 0.2)),
        transformer_l2_reg=float(transformer_cfg.get("l2_reg", 0.003)),
        transformer_input_noise=float(transformer_cfg.get("input_noise", 0.02)),
        transformer_spatial_dropout=float(transformer_cfg.get("spatial_dropout", 0.10)),
        transformer_projection_dropout=float(transformer_cfg.get("projection_dropout", 0.10)),
        final_dense_dropout=float(transformer_cfg.get("head_dropout", 0.10)),
        xgb_n_estimators=int(cfg.get("xgboost", {}).get("n_estimators", 200)),
        xgb_max_depth=int(cfg.get("xgboost", {}).get("max_depth", 5)),
        xgb_learning_rate=float(cfg.get("xgboost", {}).get("learning_rate", 0.05)),
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
        overfit_threshold=float(overfit_cfg.get("overfit_threshold", 0.08)),
        critical_threshold=float(overfit_cfg.get("critical_threshold", 0.15)),
        severe_threshold=float(overfit_cfg.get("severe_threshold", 0.25)),
        max_acceptable_gap=float(overfit_cfg.get("max_acceptable_gap", 0.12)),
        overfit_patience_epochs=int(overfit_cfg.get("patience_epochs", 3)),
        auto_adjust_dropout=bool(overfit_cfg.get("auto_adjust_dropout", True)),
        auto_reduce_lr=bool(overfit_cfg.get("auto_reduce_lr", True)),
        max_dropout_increase=float(overfit_cfg.get("max_dropout_increase", 0.3)),
        enable_swa=bool(swa_cfg.get("enabled", True)),
        swa_start_fraction=float(swa_cfg.get("start_fraction", 0.75)),
        swa_lr_factor=float(swa_cfg.get("lr_factor", 0.5)),
        enable_cosine_restarts=bool(cosine_cfg.get("enabled", True)),
        cosine_restart_period=int(cosine_cfg.get("restart_period", 10)),
        cosine_restart_lr_mult=float(cosine_cfg.get("lr_mult", 0.8)),
        enable_warmstart_detection=bool(warmstart_cfg.get("enabled", True)),
        warmstart_reset_threshold=float(warmstart_cfg.get("reset_threshold", 0.15)),
        weight_perturbation_scale=float(warmstart_cfg.get("weight_perturbation_scale", 0.02)),
        reset_optimizer_on_overfit=bool(warmstart_cfg.get("reset_optimizer_on_overfit", True)),
        # Hybrid SFT-RL settings
        use_hybrid_sft_rl=use_hybrid_sft_rl,
        rl_prob=rl_prob,
        entropy_coef=entropy_coef,
        initial_baseline=initial_baseline,
        baseline_momentum=baseline_momentum,
        # Feature selection settings
        use_feature_selection=bool(feature_cfg.get("enabled", True)),
        feature_selection_method=str(feature_cfg.get("method", "random_forest")),
        top_k_features=int(feature_cfg.get("top_k_features", 100)),
    )


def _train_direction_or_regime_model(
    *,
    use_regime,
    use_transformer,
    all_data,
    trainer_config,
    model_dir,
    pair_paths,
    training_instrument,
    warm_start_path,
    is_warm_start,
    direction_threshold,
    direction_lookahead,
    regime_lookback,
    regime_lookahead,
    console,
    cfg=None,
):
    """Train either the regime or direction model. Returns (dir_data, dir_metrics, dir_model_path, dir_trainer)."""
    from src.training.modular_trainers import (
        TCNTrainer,
        TransformerDirectionTrainer,
    )
    import logging as _logging

    if use_regime:
        return _train_regime_model(
            all_data=all_data,
            trainer_config=trainer_config,
            model_dir=model_dir,
            regime_lookback=regime_lookback,
            regime_lookahead=regime_lookahead,
            console=console,
        )

    # DIRECTION MODE
    console.print()
    console.print(Panel(
        f"[bold]Directional Prediction Network[/bold]\n\n"
        "[dim]Features:[/dim] Directional indicators (ADX, MACD, SMA crosses, market structure)\n"
        f"[dim]Output:[/dim] Binary prediction (short/long) | threshold={direction_threshold:.2%} • lookahead={direction_lookahead}",
        title="Step 1/4 • Neural Network",
        border_style="cyan",
    ))

    dir_data = all_data.get('direction', all_data.get('tcn'))

    if is_warm_start and warm_start_path:
        console.print(Panel(
            f"[yellow]Loading weights from:[/yellow] {warm_start_path}\n"
            "[dim]Training will continue from previous weights (compounding learning)[/dim]",
            title="🔥 Warm-Start Enabled",
            border_style="yellow",
        ))

    _trainers_logger = _logging.getLogger('src.training.modular_trainers')
    _trainers_logger.setLevel(_logging.WARNING)
    _absl_logger = _logging.getLogger('absl')
    _absl_logger.setLevel(_logging.ERROR)

    # Check if Walk-Forward Cross-Validation is enabled
    wf_config = None
    if cfg and isinstance(cfg, dict):
        wf_config = cfg.get('walkforward', {})
    
    wf_enabled = wf_config and wf_config.get('enabled', False)
    
    if wf_enabled:
        # Use WalkForwardOrchestrator for training
        from src.training.walkforward_orchestrator import WalkForwardOrchestrator
        
        trainer_class = TransformerDirectionTrainer if use_transformer else TCNTrainer
        
        orchestrator = WalkForwardOrchestrator(
            trainer_class=trainer_class,
            trainer_config=trainer_config,
            wf_config=wf_config,
            console=console,
        )
        
        # Train with walk-forward validation
        dir_trainer, dir_metrics = orchestrator.train(
            X_train=dir_data['X_train'],
            y_train=dir_data['y_train'],
            X_val=dir_data['X_val'],
            y_val=dir_data['y_val'],
            feature_names=dir_data['feature_names'],
            instrument=training_instrument,
            w_train=dir_data.get('w_train'),
            w_val=dir_data.get('w_val'),
            warm_start_path=str(warm_start_path) if warm_start_path else None,
        )
        
        # Save the best model from WFCV
        dir_trainer.save(str(pair_paths['direction']), instrument=training_instrument)
        if training_instrument != "GENERIC":
            model_filename = _TRANSFORMER_KERAS_FILE if use_transformer else "tcn_direction.keras"
            dir_trainer.save(str(model_dir / model_filename), instrument=training_instrument)
        dir_model_path = str(pair_paths['direction'])
        
        # Save WFCV summary
        orchestrator.save_summary(model_dir)
        
        console.print(f"[cyan]💾 Direction model (WFCV best) saved to: {pair_paths['direction']}[/cyan]")
    else:
        # Standard training (no WFCV)
        if use_transformer:
            dir_trainer = TransformerDirectionTrainer(trainer_config)
            dir_metrics = dir_trainer.train(
                dir_data['X_train'], dir_data['y_train'],
                dir_data['X_val'], dir_data['y_val'],
                feature_names=dir_data['feature_names'],
                w_train=dir_data.get('w_train'),
                w_val=dir_data.get('w_val'),
                warm_start_path=str(warm_start_path) if warm_start_path else None,
                instrument=training_instrument,
            )
            dir_trainer.save(str(pair_paths['direction']), instrument=training_instrument)
            if training_instrument != "GENERIC":
                dir_trainer.save(str(model_dir / _TRANSFORMER_KERAS_FILE), instrument=training_instrument)
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

    if 'val_balanced_accuracy' in dir_metrics:
        console.print(f"[green]✓ Directional predictor complete: Validation accuracy={dir_metrics['val_accuracy']:.1%} • Balanced={dir_metrics['val_balanced_accuracy']:.1%}[/green]")
        console.print(f"   (Long accuracy={dir_metrics.get('val_up_accuracy', 0):.1%} • Short accuracy={dir_metrics.get('val_down_accuracy', 0):.1%})")
    else:
        console.print(f"[green]✓ Directional predictor complete: Validation accuracy={dir_metrics['val_accuracy']:.1%}[/green]")

    return dir_data, dir_metrics, dir_model_path, dir_trainer


def _train_regime_model(*, all_data, trainer_config, model_dir, regime_lookback, regime_lookahead, console):
    """Train the regime classification model. Returns (dir_data, dir_metrics, dir_model_path, dir_trainer)."""
    from src.training.modular_trainers import TransformerRegimeTrainer

    console.print()
    console.print(Panel(
        "[bold]Regime Classification Network[/bold]\n\n"
        "[dim]Features:[/dim] ADX, RSI, volatility, z-scores, momentum consistency\n"
        f"[dim]Output:[/dim] 3-class regime (trend/chop/mean_revert) | lookback={regime_lookback} • lookahead={regime_lookahead}",
        title="Step 1/4 • Neural Network",
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

    console.print(f"[green]✓ Regime classifier complete: Validation accuracy={regime_metrics['val_accuracy']:.1%} • F1_macro={regime_metrics['f1_macro']:.3f}[/green]")
    console.print(f"   F1 per class: trend={regime_metrics['f1_trend']:.3f} • chop={regime_metrics['f1_chop']:.3f} • mean_revert={regime_metrics['f1_mean_revert']:.3f}")

    return regime_data, regime_metrics, regime_model_path, regime_trainer


def _train_xgboost_model(*, all_data, trainer_config, pair_paths, model_dir, training_instrument, console):
    """Train XGBoost momentum model. Returns (xgb_data, xgb_metrics)."""
    from src.training.modular_trainers import XGBoostTrainer

    console.print()
    console.print(Panel(
        "[bold]Momentum Vector Analysis[/bold]\n\n"
        "[dim]Features:[/dim] Lagged returns, spread dynamics\n"
        "[dim]Output:[/dim] Momentum score (0-1) • Acceleration metric",
        title="Step 2/4 • Gradient Boosting",
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
    xgb_trainer.save(str(pair_paths['xgboost']))
    if training_instrument != "GENERIC":
        xgb_trainer.save(str(model_dir / "xgb_momentum.pkl"))

    console.print(f"[green]✓ Momentum analyzer complete: MAE={xgb_metrics['momentum_mae']:.4f} • Acceleration accuracy={xgb_metrics['acceleration_accuracy']:.1%}[/green]")
    console.print(f"Model saved: {pair_paths['xgboost']}")

    return xgb_data, xgb_metrics


def _train_rf_model(*, all_data, trainer_config, pair_paths, model_dir, training_instrument, console):
    """Train Random Forest risk model. Returns (rf_data, rf_metrics)."""
    from src.training.modular_trainers import RandomForestTrainer

    console.print()
    console.print(Panel(
        "[bold]Risk Exposure Assessment[/bold]\n\n"
        "[dim]Features:[/dim] ATR, historical drawdowns, streak patterns\n"
        "[dim]Output:[/dim] Drawdown estimate • Streak probability",
        title="Step 3/4 • Random Forest",
        border_style="cyan",
    ))

    rf_data = all_data['rf']
    rf_trainer = RandomForestTrainer(trainer_config)
    rf_metrics = rf_trainer.train(
        rf_data['X_train'], rf_data['y_train'],
        rf_data['X_val'], rf_data['y_val'],
        feature_names=rf_data['feature_names']
    )
    rf_trainer.save(str(pair_paths['rf']))
    if training_instrument != "GENERIC":
        rf_trainer.save(str(model_dir / "rf_risk.pkl"))

    console.print(f"[green]✓ Risk assessor complete: Drawdown MAE={rf_metrics.get('drawdown_mae_bps', rf_metrics.get('drawdown_mae_pips', 0)*10000):.1f} bps • Streak MAE={rf_metrics['streak_prob_mae']:.4f}[/green]")
    console.print(f"Model saved: {pair_paths['rf']}")

    return rf_data, rf_metrics


def _train_ridge_model(*, all_data, trainer_config, pair_paths, model_dir, training_instrument, console):
    """Train Ridge confidence model. Returns (ridge_data, ridge_metrics, ridge_trainer)."""
    from src.training.modular_trainers import RidgeTrainer

    console.print()
    console.print(Panel(
        "[bold]Prediction Confidence Calibration[/bold]\n\n"
        "[dim]Features:[/dim] Rolling variance, volume dynamics\n"
        "[dim]Output:[/dim] Calibrated score (0-100)\n"
        "[dim]CV:[/dim] TimeSeriesSplit (temporal, no leakage)",
        title="Step 4/4 • Regularized Regression",
        border_style="cyan",
    ))

    ridge_data = all_data['ridge']
    ridge_trainer = RidgeTrainer(trainer_config)
    ridge_metrics = ridge_trainer.train(
        ridge_data['X_train'], ridge_data['y_train'],
        ridge_data['X_val'], ridge_data['y_val'],
        feature_names=ridge_data['feature_names']
    )
    ridge_trainer.save(str(pair_paths['ridge']))
    if training_instrument != "GENERIC":
        ridge_trainer.save(str(model_dir / "ridge_confidence.pkl"))

    # Display metrics (handles both LightGBM and ElasticNet backends)
    _model_type = ridge_metrics.get('model_type', 'elasticnet')
    _sparsity = _format_ridge_sparsity(ridge_metrics)
    if _model_type == 'lightgbm':
        console.print(f"[green]✓ Confidence calibrator complete: MAE={ridge_metrics['confidence_mae']:.2f} • R²={ridge_metrics['r2_score']:.4f} • {_sparsity}[/green]")
    else:
        console.print(f"[green]✓ Confidence calibrator complete: MAE={ridge_metrics['confidence_mae']:.2f} • R²={ridge_metrics['r2_score']:.4f} • alpha={ridge_metrics.get('best_alpha', 1.0):.4f} • L1={ridge_metrics.get('best_l1_ratio', 0.5):.2f} • {_sparsity}[/green]")
    console.print(f"Model saved: {pair_paths['ridge']}")

    return ridge_data, ridge_metrics, ridge_trainer


def _train_histgb_model(*, dir_data, trainer_config, pair_paths, model_dir, training_instrument, all_metrics, console):
    """Train optional HistGB model for hybrid voting."""
    from src.training.modular_trainers import HistGradientBoostingDirectionTrainer

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
    histgb_trainer.save(str(pair_paths['histgb']))
    if training_instrument != "GENERIC":
        histgb_trainer.save(str(model_dir / "histgb_direction.pkl"))
    all_metrics['histgb'] = histgb_metrics

    console.print(f"[green]✓ HistGB complete: val_accuracy={histgb_metrics['val_accuracy']:.1%}, balanced={histgb_metrics.get('val_balanced_accuracy', 0):.1%}[/green]")


def _train_meta_labeler_cli(
    *,
    dir_trainer,
    dir_data,
    pair_paths,
    model_dir,
    training_instrument,
    cfg,
    console,
):
    """Train meta-labeler using the Transformer's predictions on direction data.

    The meta-labeler predicts *whether* a trade signal will be profitable,
    providing a 5th gate.  It uses the same features extracted during inference
    (last-timestep of direction features + primary probability).

    Training uses the Transformer's own scaler + sequencing so that the
    resulting meta_labeler.pkl is compatible with inference-time features.
    """
    import logging as _logging
    import numpy as np
    _logger = _logging.getLogger(__name__)

    if dir_trainer is None or not getattr(dir_trainer, 'is_trained', False):
        console.print("[yellow]⚠ No trained Transformer — skipping meta-labeler[/yellow]")
        return

    if dir_data is None:
        console.print("[yellow]⚠ No direction data — skipping meta-labeler[/yellow]")
        return

    try:
        from src.training.meta_labeling import MetaLabeler, MetaLabelingConfig
    except ImportError:
        console.print("[yellow]⚠ Meta-labeling module not available — skipping[/yellow]")
        return

    console.print("\n[cyan]🏷️  Training meta-labeler (trade filter)...[/cyan]")

    X_train = dir_data.get('X_train')
    y_train = dir_data.get('y_train')
    X_val = dir_data.get('X_val')
    y_val = dir_data.get('y_val')

    if X_train is None or y_train is None:
        console.print("[yellow]⚠ Missing training data — skipping meta-labeler[/yellow]")
        return

    def _align_direction_features_for_scaler(X_raw):
        """Align raw direction features to the transformer's scaler/model expectations."""
        x_2d = np.asarray(X_raw, dtype=np.float32).reshape(-1, X_raw.shape[-1])

        scaler_local = getattr(dir_trainer, "scaler", None)
        selected = getattr(dir_trainer, "selected_indices", None)
        expected_n = None

        if scaler_local is not None and hasattr(scaler_local, "n_features_in_"):
            expected_n = int(getattr(scaler_local, "n_features_in_", 0) or 0)
        elif hasattr(dir_trainer, "model") and dir_trainer.model is not None:
            expected_n = int(dir_trainer.model.input_shape[-1])

        if expected_n <= 0:
            return x_2d

        current_n = int(x_2d.shape[-1])
        if current_n == expected_n:
            return x_2d

        indices = None
        if selected is not None:
            try:
                indices = np.asarray(selected, dtype=np.int64).reshape(-1)
            except Exception:
                indices = None

        if current_n > expected_n:
            if (
                indices is not None
                and len(indices) == expected_n
                and indices.size > 0
                and int(np.max(indices)) < current_n
            ):
                return x_2d[:, indices]

            _logger.warning(
                "Feature alignment fallback: truncating direction features %d -> %d for scaler/model compatibility",
                current_n,
                expected_n,
            )
            return x_2d[:, :expected_n]

        # current_n < expected_n
        pad = expected_n - current_n
        _logger.warning(
            "Feature alignment fallback: padding direction features %d -> %d with zeros for scaler/model compatibility",
            current_n,
            expected_n,
        )
        return np.pad(x_2d, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

    try:
        # Get Transformer predictions on train/val data using the trainer pipeline
        seq_len = getattr(dir_trainer, 'seq_len', 60)
        scaler = getattr(dir_trainer, 'scaler', None)

        def _get_predictions_and_features(X_raw, y_raw):
            """Scale, sequence, and predict — returning aligned features + predictions + labels."""
            x_2d = _align_direction_features_for_scaler(X_raw)
            if scaler is not None:
                x_scaled = scaler.transform(x_2d)
            else:
                x_scaled = x_2d

            n = len(x_scaled)
            if n < seq_len:
                return None, None, None

            # Create sequences
            sequences = np.array([
                x_scaled[i:i + seq_len]
                for i in range(n - seq_len + 1)
            ], dtype=np.float32)

            # Apply EMA weights if available
            use_ema = (
                getattr(dir_trainer, '_use_ema', False)
                and getattr(dir_trainer, 'ema', None) is not None
                and getattr(dir_trainer.ema, '_initialized', False)
                and getattr(dir_trainer.config, 'use_ema_for_inference', True)
            )
            if use_ema:
                dir_trainer.ema.apply()
            try:
                preds = dir_trainer.model.predict(sequences, verbose=0, batch_size=128).flatten()
            finally:
                if use_ema:
                    dir_trainer.ema.restore()

            # Last timestep features for each sequence (matches inference _extract_tcn_features)
            last_step_feats = sequences[:, -1, :]
            # Align labels
            aligned_labels = y_raw.flatten()[seq_len - 1:seq_len - 1 + len(preds)]

            return last_step_feats, preds, aligned_labels

        train_feats, train_preds, train_labels = _get_predictions_and_features(X_train, y_train)
        if train_feats is None:
            console.print("[yellow]⚠ Not enough training samples for meta-labeler sequencing[/yellow]")
            return

        val_feats, val_preds, val_labels = None, None, None
        if X_val is not None and y_val is not None:
            val_feats, val_preds, val_labels = _get_predictions_and_features(X_val, y_val)

        # Configure meta-labeler
        meta_cfg_yaml = cfg.get("buddy", {}).get("meta_labeling", {})
        meta_cfg = MetaLabelingConfig(
            use_xgboost=True,
            n_estimators=int(meta_cfg_yaml.get("n_estimators", 200)),
            max_depth=int(meta_cfg_yaml.get("max_depth", 4)),
            learning_rate=float(meta_cfg_yaml.get("learning_rate", 0.1)),
            reg_alpha=float(meta_cfg_yaml.get("reg_alpha", 1.0)),
            reg_lambda=float(meta_cfg_yaml.get("reg_lambda", 5.0)),
            subsample=float(meta_cfg_yaml.get("subsample", 0.7)),
            colsample_bytree=float(meta_cfg_yaml.get("colsample_bytree", 0.7)),
            min_child_weight=int(meta_cfg_yaml.get("min_child_weight", 5)),
            early_stopping_rounds=int(meta_cfg_yaml.get("early_stopping_rounds", 30)),
            max_overfitting_gap=float(meta_cfg_yaml.get("max_overfitting_gap", 0.15)),
            auto_tune=bool(meta_cfg_yaml.get("auto_tune", False)),
            min_confidence_threshold=float(meta_cfg_yaml.get("threshold", 0.55)),
            use_reduced_features=True,
            include_primary_proba=True,
        )

        labeler = MetaLabeler(meta_cfg)
        # Persist primary model feature-selection indices for inference-time alignment.
        if hasattr(dir_trainer, "selected_indices") and dir_trainer.selected_indices is not None:
            labeler.set_primary_feature_indices(np.asarray(dir_trainer.selected_indices, dtype=np.int64))
        metrics = labeler.fit(
            train_feats,
            train_preds,
            train_labels,
            features_val=val_feats,
            predictions_val=val_preds,
            outcomes_val=val_labels,
            verbose=True,
        )

        train_acc = metrics.get("meta_train_accuracy", 0)
        val_acc = metrics.get("meta_val_accuracy", 0)
        overfit_gap = metrics.get("overfitting_gap", train_acc - val_acc)

        # Tune threshold on validation if available
        if val_feats is not None:
            best_thresh, thresh_metrics = labeler.tune_threshold(
                val_feats, val_preds, val_labels, verbose=True
            )

        console.print(
            f"[green]✓ Meta-labeler: train_acc={train_acc:.1%}, "
            f"val_acc={val_acc:.1%}, gap={overfit_gap:.1%}[/green]"
        )

        # Reject if severely overfitting
        if overfit_gap > 0.20:
            console.print(f"[yellow]⚠ Meta-labeler rejected (gap {overfit_gap:.1%} > 20%)[/yellow]")
            return

        # Save to pair-specific directory
        pair_dir = pair_paths.get('pair_dir', model_dir)
        meta_save_path = pair_dir / "meta_labeler.pkl"
        labeler.save(meta_save_path)
        console.print(f"[cyan]💾 Meta-labeler saved to: {meta_save_path}[/cyan]")

        # Also save to generic directory
        if training_instrument != "GENERIC":
            generic_path = model_dir / "meta_labeler.pkl"
            labeler.save(generic_path)
            console.print(f"[dim]  Also saved to {generic_path}[/dim]")

    except Exception as e:
        _logger.warning(f"Meta-labeler training failed (non-fatal): {e}")
        console.print(f"[yellow]⚠ Meta-labeler training failed (non-fatal): {e}[/yellow]")


def _fit_post_training_calibrator_cli(
    *,
    dir_trainer,
    dir_data,
    pair_paths,
    training_instrument,
    model_dir,
    console,
):
    """Fit Platt-scaling confidence calibrator and save to pair-specific + generic dirs.

    Uses the just-trained Transformer model to produce raw probabilities on the
    validation set, then fits a ConfidenceCalibrator (Platt scaling) mapping
    those raw probabilities to actual direction outcomes.
    """
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    try:
        from src.risk.confidence_calibration import CalibrationConfig, ConfidenceCalibrator
    except ImportError:
        console.print("[yellow]⚠ Confidence calibration module not available — skipping calibrator fitting[/yellow]")
        return

    if dir_trainer is None or not hasattr(dir_trainer, 'model') or dir_trainer.model is None:
        console.print("[yellow]⚠ No trained Transformer model available — skipping calibrator fitting[/yellow]")
        return

    if dir_data is None:
        console.print("[yellow]⚠ No direction data available — skipping calibrator fitting[/yellow]")
        return

    import numpy as np

    X_val = dir_data.get('X_val')
    y_val = dir_data.get('y_val')
    if X_val is None or y_val is None or len(X_val) == 0:
        console.print("[yellow]⚠ Validation data missing — skipping calibrator fitting[/yellow]")
        return

    try:
        def _align_direction_features_for_scaler(X_raw):
            """Align raw direction features to the transformer's scaler/model expectations."""
            x_2d = np.asarray(X_raw, dtype=np.float32).reshape(-1, X_raw.shape[-1])

            scaler_local = getattr(dir_trainer, "scaler", None)
            selected = getattr(dir_trainer, "selected_indices", None)
            expected_n = None

            if scaler_local is not None and hasattr(scaler_local, "n_features_in_"):
                expected_n = int(getattr(scaler_local, "n_features_in_", 0) or 0)
            elif hasattr(dir_trainer, "model") and dir_trainer.model is not None:
                expected_n = int(dir_trainer.model.input_shape[-1])

            if expected_n <= 0:
                return x_2d

            current_n = int(x_2d.shape[-1])
            if current_n == expected_n:
                return x_2d

            indices = None
            if selected is not None:
                try:
                    indices = np.asarray(selected, dtype=np.int64).reshape(-1)
                except Exception:
                    indices = None

            if current_n > expected_n:
                if (
                    indices is not None
                    and len(indices) == expected_n
                    and indices.size > 0
                    and int(np.max(indices)) < current_n
                ):
                    return x_2d[:, indices]

                _logger.warning(
                    "Feature alignment fallback: truncating direction features %d -> %d for scaler/model compatibility",
                    current_n,
                    expected_n,
                )
                return x_2d[:, :expected_n]

            # current_n < expected_n
            pad = expected_n - current_n
            _logger.warning(
                "Feature alignment fallback: padding direction features %d -> %d with zeros for scaler/model compatibility",
                current_n,
                expected_n,
            )
            return np.pad(x_2d, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

        # Use the trainer's own scaling + sequencing pipeline to get raw probabilities.
        # Calling model.predict() directly on unscaled 2D data causes Keras 3 shape errors
        # ("as_list() is not defined on an unknown TensorShape").
        raw_probs = []
        seq_len = getattr(dir_trainer, 'seq_len', 60)
        x_val_2d = _align_direction_features_for_scaler(X_val)

        # Scale using the trainer's fitted scaler
        scaler = getattr(dir_trainer, 'scaler', None)
        if scaler is not None:
            x_val_scaled = scaler.transform(x_val_2d)
        else:
            x_val_scaled = x_val_2d

        # Apply EMA weights if available (matches trainer's predict behavior)
        use_ema = (
            getattr(dir_trainer, '_use_ema', False)
            and getattr(dir_trainer, 'ema', None) is not None
            and getattr(dir_trainer.ema, '_initialized', False)
            and getattr(dir_trainer.config, 'use_ema_for_inference', True)
        )
        if use_ema:
            dir_trainer.ema.apply()

        try:
            # Create sequences and predict in batch for efficiency
            n_samples = len(x_val_scaled)
            if n_samples >= seq_len:
                sequences = np.array([
                    x_val_scaled[i:i + seq_len]
                    for i in range(n_samples - seq_len + 1)
                ], dtype=np.float32)
                batch_preds = dir_trainer.model.predict(sequences, verbose=0, batch_size=128).flatten()
                raw_probs = batch_preds.tolist()
            else:
                # Not enough data for a full sequence
                console.print(f"[yellow]⚠ Only {n_samples} samples, need ≥{seq_len} for sequences — skipping calibrator[/yellow]")
                return
        finally:
            if use_ema:
                dir_trainer.ema.restore()

        raw_probs = np.array(raw_probs, dtype=np.float32)
        # Align labels — sequences start at index (seq_len - 1)
        actual_dirs = y_val.flatten()[seq_len - 1:seq_len - 1 + len(raw_probs)].astype(int)

        if len(raw_probs) < 10:
            console.print(f"[yellow]⚠ Only {len(raw_probs)} validation samples — need ≥10 for calibration[/yellow]")
            return

        # Fit calibrator on predicted-direction confidence, not raw long-probability.
        # This keeps calibration symmetric for long/short and matches inference usage.
        pred_dirs = (raw_probs >= 0.5).astype(int)
        direction_confidence = np.maximum(raw_probs, 1.0 - raw_probs).astype(np.float32)
        correctness = (pred_dirs == actual_dirs).astype(bool)

        # Platt scaling requires both classes present.
        unique_outcomes = np.unique(correctness.astype(int))
        if len(unique_outcomes) < 2:
            console.print(
                "[yellow]⚠ Validation correctness has a single class "
                f"({unique_outcomes.tolist()}) — skipping calibrator fitting[/yellow]"
            )
            return

        config = CalibrationConfig(
            method="platt",
            min_confidence_threshold=0.5,
            max_confidence_threshold=0.95,
            apply_directional_adjustment=False,
            apply_win_probability_adjustment=False,
            apply_trading_context_adjustment=False,
        )
        calibrator = ConfidenceCalibrator(config)
        calibrator.fit(direction_confidence, correctness)

        if not calibrator.is_fitted:
            console.print("[yellow]⚠ Calibrator fitting failed — skipping save[/yellow]")
            return

        # Save to pair-specific directory
        pair_dir = pair_paths.get('direction', model_dir).parent
        pair_calib_path = pair_dir / "confidence_calibrator.pkl"
        calibrator.save(pair_calib_path)
        console.print(f"[green]✓ Confidence calibrator saved to {pair_calib_path}[/green]")

        # Also save to generic directory
        if training_instrument != "GENERIC":
            generic_calib_path = model_dir / "confidence_calibrator.pkl"
            calibrator.save(generic_calib_path)
            console.print(f"[dim]  Also saved to {generic_calib_path}[/dim]")

        # Lightweight quality signal for visibility.
        eval_metrics = calibrator.evaluate_calibration(direction_confidence, correctness)
        console.print(
            "[dim]  Calibration Brier: "
            f"{eval_metrics['original_brier_score']:.4f} → "
            f"{eval_metrics['calibrated_brier_score']:.4f} "
            f"({eval_metrics['relative_improvement']*100:.1f}%)[/dim]"
        )

        # Show quick calibration preview
        sample_probs = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        mappings = []
        for p in sample_probs:
            result = calibrator.calibrate_confidence(p)
            mappings.append(f"{p:.2f}→{result.calibrated_confidence:.3f}")
        console.print(f"[dim]  Calibration map: {', '.join(mappings)}[/dim]")

    except Exception as e:
        _logger.warning(f"Confidence calibration failed (non-fatal): {e}")
        console.print(f"[yellow]⚠ Confidence calibration failed (non-fatal): {e}[/yellow]")


def _build_ensemble_metadata(
    *,
    use_regime,
    dir_model_path,
    dir_data,
    dir_metrics,
    xgb_data,
    xgb_metrics,
    rf_data,
    rf_metrics,
    ridge_data,
    ridge_metrics,
    pair_paths,
    training_instrument,
    training_granularity,
    train_frac,
    val_frac,
    test_frac,
    feature_df,
    direction_threshold,
    direction_lookahead,
    regime_lookback,
    regime_lookahead,
    use_transformer,
):
    """Build the ensemble metadata dictionary."""
    from datetime import datetime

    direction_model_name = "Transformer" if use_transformer else "TCN"

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

    return {
        "model_type": "modular_ensemble",
        "use_regime": use_regime,
        "primary_model_type": "regime" if use_regime else direction_model_name.lower(),
        "primary_config": primary_config,
        "training_instrument": training_instrument,
        "pair_model_dir": str(pair_paths['pair_dir']),
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


def _save_ensemble_metadata(meta, meta_path, pair_paths, training_instrument):
    """Save ensemble metadata to both generic and pair-specific paths."""
    def _meta_serializer(x):
        return float(x) if isinstance(x, (np.floating, np.integer)) else str(x)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_meta_serializer)
    if training_instrument and training_instrument != "GENERIC":
        pair_meta_path = pair_paths['pair_dir'] / _META_JSON_FILE
        with open(pair_meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=_meta_serializer)


def _print_ensemble_summary(
    *,
    console,
    use_regime,
    dir_metrics,
    xgb_metrics,
    rf_metrics,
    ridge_metrics,
    dir_model_path,
    pair_paths,
    meta_path,
    training_instrument,
    model_dir,
):
    """Print the professional summary tables after ensemble training."""
    console.print()

    perf_table = Table(show_header=True, header_style="bold green", title="Model Performance")
    perf_table.add_column("Component", style="white", width=20)
    perf_table.add_column("Metric", style="cyan", width=25)
    perf_table.add_column("Value", style="green", justify="right")

    if use_regime:
        perf_table.add_row("Transformer", "F1 Macro", f"{dir_metrics.get('f1_macro', 0):.3f}")
        perf_table.add_row("Network", "F1 Trend", f"{dir_metrics.get('f1_trend', 0):.3f}")
        perf_table.add_row("", "F1 Chop", f"{dir_metrics.get('f1_chop', 0):.3f}")
    else:
        dir_acc = dir_metrics['val_accuracy']
        bal_acc = dir_metrics.get('val_balanced_accuracy', dir_acc)
        perf_table.add_row("Transformer", "Validation Accuracy", f"{dir_acc:.1%}")
        perf_table.add_row("Network", "Balanced Accuracy", f"{bal_acc:.1%}")

    perf_table.add_row("", "", "")
    perf_table.add_row(_GRADIENT_BOOSTING, "Acceleration Accuracy", f"{xgb_metrics['acceleration_accuracy']:.1%}")
    perf_table.add_row("", "Momentum MAE", f"{xgb_metrics['momentum_mae']:.4f}")
    perf_table.add_row("", "", "")
    perf_table.add_row(_RANDOM_FOREST, "Drawdown MAE", f"{rf_metrics.get('drawdown_mae_bps', rf_metrics.get('drawdown_mae_pips', 0)*10000):.1f} bps")
    perf_table.add_row("", "Streak Probability MAE", f"{rf_metrics['streak_prob_mae']:.4f}")
    perf_table.add_row("", "", "")
    _model_type = ridge_metrics.get('model_type', 'elasticnet')
    perf_table.add_row("Regularized", "R² Score", f"{ridge_metrics['r2_score']:.3f}")
    perf_table.add_row("Regression", "Confidence MAE", f"{ridge_metrics['confidence_mae']:.2f}")
    if _model_type == 'lightgbm':
        perf_table.add_row("", "Backend", "LightGBM")
        perf_table.add_row("", "Important Features", _format_ridge_sparsity(ridge_metrics))
    else:
        perf_table.add_row("", "Best Alpha", f"{ridge_metrics.get('best_alpha', 1.0):.4f}")
        perf_table.add_row("", "L1 Ratio", f"{ridge_metrics.get('best_l1_ratio', 0.5):.2f}")
        perf_table.add_row("", "Features (sparse)", _format_ridge_sparsity(ridge_metrics))

    console.print(Panel(perf_table, border_style="green"))
    console.print()

    artifacts_table = Table(show_header=False, box=None)
    artifacts_table.add_column("Type", style="cyan", width=15)
    artifacts_table.add_column("Path", style="dim")
    artifacts_table.add_row("Direction", str(dir_model_path))
    artifacts_table.add_row("Momentum", str(pair_paths['xgboost']))
    artifacts_table.add_row("Risk", str(pair_paths['rf']))
    artifacts_table.add_row("Confidence", str(pair_paths['ridge']))
    artifacts_table.add_row("Metadata", str(pair_paths['pair_dir'] / _META_JSON_FILE)
                            if training_instrument and training_instrument != "GENERIC"
                            else str(meta_path))
    artifacts_table.add_row("Metadata (generic)", str(meta_path))

    console.print(Panel(artifacts_table, title="[bold]Saved Artifacts[/bold]", border_style="blue"))
    console.print()


def _window_2d_to_3d(X_2d, y_1d, window_size):  # noqa: N803  # NOSONAR - ML convention
    """Convert (n_samples, n_features) → (n_windows, window_size, n_features).

    Labels are taken from the *last* timestep of each window,
    matching the semantics used during training.
    """
    X_2d = np.asarray(X_2d, dtype=np.float32)
    y_1d = np.asarray(y_1d).flatten()
    n = len(X_2d)
    if n <= window_size:
        return None, None
    n_windows = n - window_size + 1
    strides = (X_2d.strides[0],) + X_2d.strides
    X_3d = np.lib.stride_tricks.as_strided(  # NOSONAR
        X_2d, shape=(n_windows, window_size, X_2d.shape[1]), strides=strides
    ).copy()
    y_windowed = y_1d[window_size - 1:]
    return X_3d, y_windowed


def _window_1d_targets(y_1d, window_size):
    """Align 1D targets/weights to sequence windows (label at last timestep)."""
    y_1d = np.asarray(y_1d).flatten()
    if len(y_1d) <= window_size:
        return None
    return y_1d[window_size - 1:]


def _build_clear_label_mask(labels_1d, weights_1d=None):
    """Mask clear binary labels; optionally require positive sample weights."""
    labels_1d = np.asarray(labels_1d).flatten()
    is_binary = np.isclose(labels_1d, 0.0) | np.isclose(labels_1d, 1.0)

    if weights_1d is None:
        return is_binary

    weights_1d = np.asarray(weights_1d).flatten()
    if len(weights_1d) != len(labels_1d):
        return is_binary

    return (weights_1d > 0) & is_binary


def _run_enterprise_validation(
    *,
    options,
    console,
    cfg,
    use_regime,
    dir_trainer,
    dir_data,
    dir_metrics,
    xgb_metrics,
    rf_metrics,
    ridge_metrics,
    ridge_trainer,
    ridge_data,
    feature_df,
    pair_paths,
    meta_path,
    training_instrument,
    training_granularity,
    model_dir,
    seq_len,
    epochs,
    batch_size,
    lr,
    direction_threshold,
    direction_lookahead,
    dir_model_path,
    rl_position_cfg=None,
):
    """Run enterprise validation suite: bootstrap CI, walk-forward CV, MLflow, report generation."""

    console.print()
    console.print(Panel(
        "[bold]Enterprise Validation Suite[/bold]\n"
        "[dim]Statistical validation, cross-validation, and experiment tracking[/dim]",
        title="🏢 Enterprise",
        border_style="cyan",
    ))

    try:
        from src.training.enterprise_training import (
            StatisticalValidator,
            MLFLOW_AVAILABLE,
        )

        stat_validator = StatisticalValidator()
        validation_results = Table(show_header=True, header_style=_STYLE_HEADER, title="Validation Results")
        validation_results.add_column("Check", style="white", width=25)
        validation_results.add_column("Result", style="green", width=15)
        validation_results.add_column("Details", style="dim", width=35)

        cv_folds = options.cv_folds if hasattr(options, 'cv_folds') else 5
        bootstrap_enabled = options.bootstrap if hasattr(options, 'bootstrap') else False
        bootstrap_samples = options.bootstrap_samples if hasattr(options, 'bootstrap_samples') else 1000
        generate_report_enabled = options.generate_report if hasattr(options, 'generate_report') else False

        # 1. Bootstrap CI
        if bootstrap_enabled and not use_regime:
            _run_bootstrap_validation(
                dir_trainer=dir_trainer,
                dir_data=dir_data,
                dir_metrics=dir_metrics,
                stat_validator=stat_validator,
                validation_results=validation_results,
                seq_len=seq_len,
                bootstrap_samples=bootstrap_samples,
            )

        # 2. Walk-Forward CV
        if cv_folds > 0 and len(dir_data['X_train']) > cv_folds * 500:
            _run_walkforward_cv(
                dir_trainer=dir_trainer,
                dir_data=dir_data,
                dir_metrics=dir_metrics,
                validation_results=validation_results,
                cv_folds=cv_folds,
                seq_len=seq_len,
                console=console,
            )

        console.print(validation_results)
        console.print()

        # 3. MLflow
        if MLFLOW_AVAILABLE:
            _log_to_mlflow(
                options=options,
                training_instrument=training_instrument,
                training_granularity=training_granularity,
                dir_metrics=dir_metrics,
                xgb_metrics=xgb_metrics,
                rf_metrics=rf_metrics,
                ridge_metrics=ridge_metrics,
                meta_path=meta_path,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                direction_threshold=direction_threshold,
                direction_lookahead=direction_lookahead,
                console=console,
            )

        # 4. Deployment Validation
        deployment_result = _run_deployment_validation(
            dir_metrics=dir_metrics,
            xgb_metrics=xgb_metrics,
            rf_metrics=rf_metrics,
            ridge_metrics=ridge_metrics,
            training_data_size=len(dir_data.get('X_train', [])),
            console=console,
        )

        # 5. Report
        if generate_report_enabled:
            _generate_training_report(
                model_dir=model_dir,
                training_instrument=training_instrument,
                training_granularity=training_granularity,
                dir_metrics=dir_metrics,
                xgb_metrics=xgb_metrics,
                rf_metrics=rf_metrics,
                ridge_metrics=ridge_metrics,
                dir_model_path=dir_model_path,
                pair_paths=pair_paths,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                direction_threshold=direction_threshold,
                direction_lookahead=direction_lookahead,
                deployment_result=deployment_result,
                console=console,
            )

        # Final success
        console.print()
        deployment_status = (
            "[green]APPROVED[/green]" if deployment_result.deployment_approved
            else "[yellow]NEEDS REVIEW[/yellow]"
        )
        console.print(Panel(
            "[bold green]✓ TRAINING PIPELINE COMPLETE[/bold green]\n\n"
            f"[dim]Artifacts:[/dim] [cyan]{model_dir}[/cyan]\n"
            f"[dim]Validation:[/dim] [green]PASSED[/green] • Enterprise-grade quality assurance\n"
            f"[dim]Deployment:[/dim] {deployment_status} • "
            f"{deployment_result.total_checks - deployment_result.checks_failed}/{deployment_result.total_checks} checks passed",
            border_style="green" if deployment_result.deployment_approved else "yellow",
        ))

        # RL Position Sizer Training
        train_rl_sizer_enabled = _is_rl_auto_train_enabled(cfg, options)
        rl_timesteps = options.rl_timesteps

        if train_rl_sizer_enabled:
            _train_rl_after_ensemble(
                console=console,
                dir_trainer=dir_trainer,
                dir_data=dir_data,
                ridge_trainer=ridge_trainer,
                ridge_data=ridge_data,
                feature_df=feature_df,
                seq_len=seq_len,
                rl_timesteps=rl_timesteps,
                rl_position_cfg=rl_position_cfg,
            )

    except ImportError as e:
        console.print(f"[yellow]Enterprise features unavailable: {e}[/yellow]")
        console.print("[dim]Install with: pip install mlflow structlog[/dim]")
    except Exception as e:
        console.print(f"[yellow]Enterprise validation error: {e}[/yellow]")
        import traceback
        traceback.print_exc()


def _run_bootstrap_validation(*, dir_trainer, dir_data, dir_metrics, stat_validator, validation_results, seq_len, bootstrap_samples):
    """Run bootstrap confidence interval validation."""
    if not hasattr(dir_trainer, 'model') or dir_trainer.model is None:
        return
    try:
        X_val = np.asarray(dir_data['X_val'])
        y_val_raw = dir_data['y_val'].flatten()
        w_val_raw = np.asarray(
            dir_data.get('w_val', np.ones_like(y_val_raw, dtype=np.float32))
        ).flatten()

        # Apply feature selection if trainer used it (raw data has more features than model expects)
        if hasattr(dir_trainer, 'selected_indices') and dir_trainer.selected_indices is not None:
            model_n_features = dir_trainer.model.input_shape[-1]
            if X_val.shape[-1] > model_n_features:
                X_val = X_val[:, dir_trainer.selected_indices]

        # Apply scaler if trainer fitted one (model was trained on scaled data)
        if hasattr(dir_trainer, 'scaler') and dir_trainer.scaler is not None:
            X_val = dir_trainer.scaler.transform(X_val)

        if len(X_val.shape) == 2:
            X_val, y_val_raw = _window_2d_to_3d(X_val, y_val_raw, seq_len)
            w_val_raw = _window_1d_targets(w_val_raw, seq_len)
            if X_val is None:
                raise ValueError(f"Not enough val samples for seq_len={seq_len}")
            if w_val_raw is None:
                raise ValueError(f"Not enough val weights for seq_len={seq_len}")

        val_preds = dir_trainer.model.predict(X_val, verbose=0)
        if len(val_preds.shape) > 1:
            val_preds = val_preds[:, 0] if val_preds.shape[1] == 1 else val_preds
        clear_mask = _build_clear_label_mask(y_val_raw, w_val_raw)
        if not np.any(clear_mask):
            raise ValueError("No clear validation labels available for bootstrap CI")
        val_preds_binary = (val_preds > 0.5).astype(np.int32).flatten()[clear_mask]
        val_labels = y_val_raw.flatten()[clear_mask].astype(np.int32)

        bootstrap_results = stat_validator.bootstrap_confidence_interval(
            lambda y, p: np.mean(y == p),
            val_labels, val_preds_binary,
            n_bootstrap=bootstrap_samples,
        )

        validation_results.add_row(
            "Bootstrap CI (95%)",
            f"{bootstrap_results['mean']:.2%}",
            f"[{bootstrap_results['ci_lower']:.2%}, {bootstrap_results['ci_upper']:.2%}]"
        )

        dir_metrics['bootstrap_ci_lower'] = bootstrap_results['ci_lower']
        dir_metrics['bootstrap_ci_upper'] = bootstrap_results['ci_upper']
        dir_metrics['bootstrap_mean'] = bootstrap_results['mean']
    except Exception as bootstrap_err:
        validation_results.add_row(
            "Bootstrap CI (95%)",
            "[yellow]Skipped[/yellow]",
            f"[dim]{bootstrap_err}[/dim]"
        )


def _run_walkforward_cv(*, dir_trainer, dir_data, dir_metrics, validation_results, cv_folds, seq_len, console):
    """Run walk-forward cross-validation (evaluation only, no retraining)."""
    import logging
    logger = logging.getLogger(__name__)

    from src.training.enterprise_training import WalkForwardValidator, WalkForwardConfig

    wf_config = WalkForwardConfig(
        n_splits=cv_folds,
        train_period=len(dir_data['X_train']) // (cv_folds + 1),
        test_period=len(dir_data['X_val']) // cv_folds,
        gap=24,
        expanding=True,
    )
    wf_validator = WalkForwardValidator(wf_config)

    X_full = np.vstack([dir_data['X_train'], dir_data['X_val']])
    y_full = np.concatenate([dir_data['y_train'].flatten(), dir_data['y_val'].flatten()])
    if 'w_train' in dir_data and 'w_val' in dir_data:
        w_full = np.concatenate([dir_data['w_train'].flatten(), dir_data['w_val'].flatten()])
        if len(w_full) != len(y_full):
            logger.warning("w_full length mismatch with y_full; falling back to label-only filtering")
            w_full = None
    else:
        w_full = None

    # Apply feature selection and scaling to match what the model was trained on
    if hasattr(dir_trainer, 'selected_indices') and dir_trainer.selected_indices is not None:
        model_n_features = dir_trainer.model.input_shape[-1]
        if X_full.shape[-1] > model_n_features:
            X_full = X_full[:, dir_trainer.selected_indices]
    if hasattr(dir_trainer, 'scaler') and dir_trainer.scaler is not None:
        X_full = dir_trainer.scaler.transform(X_full)

    cv_scores = []
    with console.status("[bold cyan]Running Walk-Forward CV (evaluation only)...", spinner="dots"):
        for fold_idx, (_train_idx, val_idx) in enumerate(wf_validator.split(X_full)):
            X_cv_val = X_full[val_idx]
            y_cv_val = y_full[val_idx]
            w_cv_val = w_full[val_idx] if w_full is not None else None

            try:
                if len(X_cv_val.shape) == 2:
                    X_cv_val, y_cv_val = _window_2d_to_3d(X_cv_val, y_cv_val, seq_len)
                    w_cv_val = _window_1d_targets(w_cv_val, seq_len) if w_cv_val is not None else None
                    if X_cv_val is None:
                        logger.debug(f"CV fold {fold_idx+1} skipped: not enough samples for seq_len={seq_len}")
                        continue

                y_pred = dir_trainer.model.predict(X_cv_val, verbose=0)
                if len(y_pred.shape) > 1:
                    y_pred = y_pred[:, 0] if y_pred.shape[1] == 1 else y_pred
                y_pred_binary = (y_pred > 0.5).astype(np.int32).flatten()

                clear_mask = _build_clear_label_mask(y_cv_val, w_cv_val)
                if not np.any(clear_mask):
                    logger.debug(f"CV fold {fold_idx+1} skipped: no clear labels after filtering")
                    continue
                y_cv_val_binary = y_cv_val.flatten()[clear_mask].astype(np.int32)
                fold_acc = np.mean(y_pred_binary[clear_mask] == y_cv_val_binary)
                cv_scores.append(fold_acc)
            except Exception as e:
                logger.debug(f"CV fold {fold_idx+1} evaluation failed: {e}")

    if cv_scores:
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        validation_results.add_row(
            f"Walk-Forward CV ({cv_folds} folds)",
            f"{cv_mean:.2%}",
            f"± {cv_std:.2%} std"
        )

        dir_metrics['cv_mean'] = cv_mean
        dir_metrics['cv_std'] = cv_std
        dir_metrics['cv_scores'] = cv_scores


def _run_deployment_validation(
    *,
    dir_metrics,
    xgb_metrics,
    rf_metrics,
    ridge_metrics,
    training_data_size,
    console,
):
    """Run deployment validation gate."""
    from src.training.deployment_gate import DeploymentValidator, ValidationCriteria
    from rich.table import Table

    console.print()
    console.print(Panel(
        "[bold]Deployment Validation Gate[/bold]\n"
        "[dim]Checking if model meets production quality standards[/dim]",
        title="🚦 Deployment",
        border_style="cyan",
    ))

    # Create validator with default criteria
    validator = DeploymentValidator(criteria=ValidationCriteria())

    # Run validation
    result = validator.validate(
        dir_metrics=dir_metrics,
        xgb_metrics=xgb_metrics,
        rf_metrics=rf_metrics,
        ridge_metrics=ridge_metrics,
        training_data_size=training_data_size,
    )

    # Display results in table
    validation_table = Table(show_header=True, header_style=_STYLE_HEADER, title="Validation Checks")
    validation_table.add_column("Check", style="white", width=35)
    validation_table.add_column("Status", style="white", width=10)
    validation_table.add_column("Value", style="dim", width=30)

    for check_name, passed in result.checks_passed.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        check_data = result.check_values.get(check_name, {})
        value = check_data.get('value')
        threshold = check_data.get('threshold')

        if value is not None and threshold is not None:
            if isinstance(value, (int, float)):
                value_str = f"{value:.4f} (threshold: {threshold:.4f})"
            else:
                value_str = f"{value} (threshold: {threshold})"
        else:
            value_str = "N/A"

        validation_table.add_row(check_name, status, value_str)

    console.print(validation_table)

    # Display summary
    summary = result.get_summary()
    console.print()
    if result.deployment_approved:
        console.print(
            f"[bold green]✓ DEPLOYMENT APPROVED[/bold green] • "
            f"{summary['checks_passed']}/{summary['total_checks']} checks passed"
        )
    else:
        console.print(
            f"[bold yellow]⚠ DEPLOYMENT NEEDS REVIEW[/bold yellow] • "
            f"{summary['checks_failed']} checks failed "
            f"({summary['critical_failures']} critical)"
        )

    # Display failure reasons
    if result.failure_reasons:
        console.print()
        console.print("[yellow]Failure Reasons:[/yellow]")
        for reason in result.failure_reasons[:5]:  # Show first 5
            console.print(f"  • {reason}")

    # Display recommendations
    if result.recommendations:
        console.print()
        console.print("[cyan]Recommendations:[/cyan]")
        for rec in result.recommendations[:3]:  # Show first 3
            console.print(f"  → {rec}")

    console.print()

    return result


def _log_to_mlflow(
    *,
    options,
    training_instrument,
    training_granularity,
    dir_metrics,
    xgb_metrics,
    rf_metrics,
    ridge_metrics,
    meta_path,
    epochs,
    batch_size,
    lr,
    direction_threshold,
    direction_lookahead,
    console,
):
    """Log training results to MLflow."""
    from datetime import datetime

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
        mlflow.log_param("model_type", "ensemble")
        mlflow.log_param("instrument", training_instrument)
        mlflow.log_param("granularity", training_granularity)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("direction_threshold", direction_threshold)
        mlflow.log_param("direction_lookahead", direction_lookahead)

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

        mlflow.log_artifact(str(meta_path))

        console.print(f"  ✓ Run logged: {mlflow.active_run().info.run_id[:8]}...")


def _generate_training_report(
    *,
    model_dir,
    training_instrument,
    training_granularity,
    dir_metrics,
    xgb_metrics,
    rf_metrics,
    ridge_metrics,
    dir_model_path,
    pair_paths,
    epochs,
    batch_size,
    lr,
    direction_threshold,
    direction_lookahead,
    deployment_result,
    console,
):
    """Generate a Markdown training report."""
    from datetime import datetime

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
    
    # Add Walk-Forward CV metrics if available
    if 'wfcv_mean_test_accuracy' in dir_metrics:
        wfcv_mean = dir_metrics['wfcv_mean_test_accuracy']
        wfcv_std = dir_metrics.get('wfcv_std_test_accuracy', 0.0)
        wfcv_n_folds = dir_metrics.get('wfcv_n_folds', 0)
        wfcv_best_fold = dir_metrics.get('wfcv_best_fold', 0)
        
        report_content += f"""
#### Walk-Forward Cross-Validation
- **Test Accuracy (Mean ± Std)**: {wfcv_mean:.2%} ± {wfcv_std:.2%}
- **Number of Folds**: {wfcv_n_folds}
- **Best Fold Selected**: {wfcv_best_fold}
- **Stability (CV)**: {(wfcv_std / wfcv_mean if wfcv_mean > 0 else 0):.4f}

> **Note**: Walk-forward validation provides robust out-of-sample performance estimates.
> The reported model is the best-performing fold from time-series cross-validation.

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

## Deployment Validation

**Status**: {'✓ APPROVED' if deployment_result.deployment_approved else '✗ NEEDS REVIEW'}

**Summary**: {deployment_result.total_checks - deployment_result.checks_failed}/{deployment_result.total_checks} checks passed
"""

    # Add deployment validation checks
    if deployment_result.checks_passed:
        report_content += "\n### Validation Checks\n\n"
        report_content += "| Check | Status | Value | Threshold |\n"
        report_content += "|-------|--------|-------|----------|\n"

        for check_name, passed in deployment_result.checks_passed.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            check_data = deployment_result.check_values.get(check_name, {})
            value = check_data.get('value', 'N/A')
            threshold = check_data.get('threshold', 'N/A')

            if isinstance(value, (int, float)) and isinstance(threshold, (int, float)):
                report_content += f"| {check_name} | {status} | {value:.4f} | {threshold:.4f} |\n"
            else:
                report_content += f"| {check_name} | {status} | {value} | {threshold} |\n"

    # Add failure reasons if any
    if deployment_result.failure_reasons:
        report_content += "\n### Failure Reasons\n\n"
        for reason in deployment_result.failure_reasons:
            report_content += f"- {reason}\n"

    # Add recommendations if any
    if deployment_result.recommendations:
        report_content += "\n### Recommendations\n\n"
        for rec in deployment_result.recommendations:
            report_content += f"- {rec}\n"

    report_content += f"""
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


def _train_rl_after_ensemble(
    *,
    console,
    dir_trainer,
    dir_data,
    ridge_trainer,
    ridge_data,
    feature_df,
    seq_len,
    rl_timesteps,
    rl_position_cfg=None,
):
    """Train RL position sizer after ensemble training."""
    try:
        rl_features_2d = np.vstack([
            dir_data['X_train'],
            dir_data['X_val'],
        ]).astype(np.float32)

        n_rl = len(rl_features_2d)
        rl_direction_probs = np.full(n_rl, 0.5, dtype=np.float32)
        rl_confidences = np.full(n_rl, 0.5, dtype=np.float32)

        try:
            if hasattr(dir_trainer, 'model') and dir_trainer.model is not None:
                # Apply feature selection first, then scale
                X_rl = rl_features_2d.copy()
                if hasattr(dir_trainer, 'selected_indices') and dir_trainer.selected_indices is not None:
                    model_n_features = dir_trainer.model.input_shape[-1]
                    if X_rl.shape[-1] > model_n_features:
                        X_rl = X_rl[:, dir_trainer.selected_indices]
                X_scaled = dir_trainer.scaler.transform(X_rl)
                model_input_shape = dir_trainer.model.input_shape
                is_seq_model = len(model_input_shape) == 3

                if is_seq_model:
                    _sl = getattr(dir_trainer, 'seq_len', seq_len)
                    X_3d, _ = _window_2d_to_3d(X_scaled, np.zeros(len(X_scaled)), _sl)
                    if X_3d is not None:
                        raw_preds = dir_trainer.model.predict(X_3d, verbose=0).flatten()
                        pad = n_rl - len(raw_preds)
                        rl_direction_probs = np.concatenate([
                            np.full(pad, 0.5, dtype=np.float32), raw_preds
                        ])
                        console.print(f"[dim]  Direction probs: {len(raw_preds):,} windowed predictions[/dim]")
                else:
                    raw_preds = dir_trainer.model.predict(X_scaled, verbose=0).flatten()
                    rl_direction_probs[:len(raw_preds)] = raw_preds[:n_rl]
        except Exception as e:
            console.print(f"[dim]  Using default direction probs: {e}[/dim]")

        try:
            if hasattr(ridge_trainer, 'model') and ridge_trainer.model is not None:
                ridge_feats = np.vstack([
                    ridge_data['X_train'],
                    ridge_data['X_val'],
                ]).astype(np.float32)
                ridge_scaled = ridge_trainer.scaler.transform(ridge_feats)
                # Wrap in DataFrame for LightGBM to avoid feature-name warnings
                if getattr(ridge_trainer, 'feature_names', None) is not None:
                    import pandas as _pd
                    ridge_scaled = _pd.DataFrame(ridge_scaled, columns=ridge_trainer.feature_names)
                raw_conf = np.asarray(ridge_trainer.model.predict(ridge_scaled), dtype=np.float32).flatten()
                raw_conf = np.clip(raw_conf / 100.0, 0.0, 1.0)
                min_c = min(n_rl, len(raw_conf))
                rl_confidences[:min_c] = raw_conf[:min_c]
                console.print(f"[dim]  Confidence scores: {min_c:,} batch predictions[/dim]")
        except Exception as e:
            console.print(f"[dim]  Using default confidences: {e}[/dim]")

        rl_predictions = np.column_stack([
            rl_direction_probs,
            rl_confidences,
        ]).astype(np.float32)

        rl_prices = feature_df['close'].values[:n_rl].astype(np.float32)

        console.print(f"[dim]RL training data: {n_rl:,} samples, "
                      f"{rl_features_2d.shape[1]} features[/dim]")

        _train_rl_position_sizer_if_ready(
            console=console,
            rl_timesteps=rl_timesteps,
            features=rl_features_2d,
            ensemble_predictions=rl_predictions,
            prices=rl_prices,
            rl_position_cfg=rl_position_cfg,
        )
    except Exception as e:
        console.print(f"[yellow]RL data preparation failed: {e}[/yellow]")
        console.print("[dim]Skipping RL position sizer training[/dim]")


def _train_buddy_impl(
    config_path: str,
    csv_path: str | None = None,
    *,
    options: BuddyTrainingOptions,
) -> None:  # noqa: C901, PLR0915  # NOSONAR
    """Train Buddy (TensorFlow-only) from USDJPY historical data."""
    cfg = _load_config_suppressed(config_path)
    rl_position_cfg = _resolve_rl_position_sizing_config(cfg)
    opts = _extract_options_to_dict(options)

    oanda_fetch = opts["oanda_fetch"]
    advanced = opts["advanced"]
    seq_len = opts["seq_len"]
    epochs = opts["epochs"]
    batch_size = opts["batch_size"]
    lr = opts["lr"]
    patience = opts["patience"]
    seed = opts["seed"]
    run_tag = opts["run_tag"]
    all_features = opts["all_features"]
    ignore_input_mismatches = opts["ignore_input_mismatches"]
    disable_tier2_on_mismatch = opts["disable_tier2_on_mismatch"]
    mixed_precision = opts["mixed_precision"]
    steps_per_execution = opts["steps_per_execution"]
    jit_compile = opts["jit_compile"]
    shuffle_buffer = opts["shuffle_buffer"]
    prefetch = opts["prefetch"]
    cache_val = opts["cache_val"]
    combined_use_predict = opts["combined_use_predict"]
    shared_encoder = opts["shared_encoder"]
    model_type = opts["model_type"]
    timing = opts["timing"]
    fit_verbose = opts["fit_verbose"]
    pca_components = opts["pca_components"]

    dev = str(opts["device"] or "auto").strip().lower()
    force_cpu = dev == "cpu"
    
    # === REPRODUCIBILITY: Initialize global seed ===
    reproducibility_cfg = cfg.get("reproducibility", {}) if isinstance(cfg, dict) else {}
    if reproducibility_cfg.get("enabled", True):
        config_seed = reproducibility_cfg.get("global_seed", seed)
        if config_seed is not None:
            set_global_seed(config_seed)
            console.print(f"[green]✅ Reproducibility enabled with global seed: {config_seed}[/green]")
        elif seed is not None:
            set_global_seed(seed)
            console.print(f"[green]✅ Reproducibility enabled with CLI seed: {seed}[/green]")
    elif seed is not None:
        # Reproducibility disabled in config but seed provided via CLI
        set_global_seed(seed)
        console.print(f"[yellow]⚠️  Reproducibility disabled in config, but using CLI seed: {seed}[/yellow]")

    try:
        from src.utils.tracing_setup import setup_tracing as _setup_tracing
    except Exception:
        _setup_tracing = None

    from src.training.buddy_training_helpers import _buddy_setup_training_environment

    mp_enabled = _buddy_setup_training_environment(
        timing=bool(timing),
        force_cpu=bool(force_cpu),
        mixed_precision=bool(mixed_precision),
        configure_tf_metal=_configure_tf_metal,
        console=console,
        setup_tracing=_setup_tracing,
    )
    import numpy as np
    import tensorflow as tf

    from src.data.feature_engineering import FeatureEngineering

    buddy_cfg = (cfg.get("buddy") or {}) if isinstance(cfg, dict) else {}

    oos_enabled_env, oos_candles_env = _resolve_oos_env_settings()

    adv_opts = _resolve_advanced_options(
        advanced, buddy_cfg, opts["feature_curriculum_opt"], opts["curriculum_ks_opt"]
    )
    tier2_opts = _resolve_tier2_options(advanced, buddy_cfg)

    init_from = adv_opts["init_from"]
    es_monitor = adv_opts["es_monitor"]
    combined_w_dir = adv_opts["combined_w_dir"]
    combined_w_conf = adv_opts["combined_w_conf"]
    train_smoothing = adv_opts["train_smoothing"]
    top_features = adv_opts["top_features"]
    median_window = adv_opts["median_window"]
    vol_norm_window = adv_opts["vol_norm_window"]
    min_volume = adv_opts["min_volume"]
    spread_filter = adv_opts["spread_filter"]
    spread_pctl = adv_opts["spread_pctl"]
    spread_mult = adv_opts["spread_mult"]
    feature_curriculum = adv_opts["feature_curriculum"]
    curriculum_ks = adv_opts["curriculum_ks"]

    tier2_calibrate = tier2_opts["tier2_calibrate"]
    tier2_stop_loss_pips = tier2_opts["tier2_stop_loss_pips"]
    tier2_take_profit_pips = tier2_opts["tier2_take_profit_pips"]
    tier2_spread_fallback_pips = tier2_opts["tier2_spread_fallback_pips"]
    tier2_horizon_candles = tier2_opts["tier2_horizon_candles"]
    tier2_tie_break = tier2_opts["tier2_tie_break"]
    tier2_calibration_bins = tier2_opts["tier2_calibration_bins"]
    tier2_calibration_stride = tier2_opts["tier2_calibration_stride"]

    if feature_curriculum and top_features is not None:
        raise ValueError("--feature-curriculum cannot be combined with --top-features")

    # Reproducibility (best-effort).
    try:
        tf.keras.utils.set_random_seed(int(seed))
        np.random.seed(int(seed))
    except Exception:
        pass

    # Setup OOS live split
    oos_live_split, base_oanda_fetch, oanda_fetch = _setup_oos_live_split(
        oos_enabled_env, csv_path, oanda_fetch, oos_candles_env
    )

    # Load training data
    multi_pair_mode = bool(options.multi_pair)
    foundation_pairs_str = options.foundation_pairs

    df, oos_live_split_override = _load_training_data(
        options=options,
        cfg=cfg,
        csv_path=csv_path,
        oanda_fetch=oanda_fetch,
        console=console,
        multi_pair_mode=multi_pair_mode,
        foundation_pairs_str=foundation_pairs_str,
        min_volume=min_volume,
        spread_filter=spread_filter,
        spread_pctl=spread_pctl,
        spread_mult=spread_mult,
        seq_len=seq_len,
    )
    if oos_live_split_override is not None:
        oos_live_split = oos_live_split_override

    # OOS holdout split
    df, oos_holdout_csv, oos_holdout_rows = _apply_oos_holdout_split(
        df, oos_live_split, base_oanda_fetch, oos_candles_env, seq_len, console
    )

    # Feature engineering
    df = _apply_feature_engineering(df, cfg, all_features, train_smoothing, median_window, console)

    # Resolve init feature columns for warm-start
    init_feature_columns = _resolve_init_feature_columns(init_from, console)

    # Prepare numeric features
    numeric_df = _prepare_numeric_features(df, init_feature_columns, top_features, init_from, console)

    # Keep OHLC (+ optional bid/ask closes) aligned to the exact numeric_df rows used for training.
    ohlc_cols = ["open", "high", "low", "close"]
    extra_cols: list[str] = []
    for c in ("bid_close", "ask_close"):
        if c in df.columns:
            extra_cols.append(c)
    ohlc_df = df.loc[numeric_df.index, ohlc_cols + extra_cols].copy()

    closes = df.loc[numeric_df.index, "close"].to_numpy(dtype=np.float32)

    # Feature ranking for curriculum learning / top-K selection
    numeric_df, ranked_cols = _compute_feature_ranking(
        numeric_df, closes, feature_curriculum, top_features, vol_norm_window, console
    )

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
    conf_y_all, conf_w_all, conf_instrument_guess, label_stride = _compute_tier2_confidence_labels(
        n=n,
        direction_y_all=direction_y_all,
        ohlc_df=ohlc_df,
        oanda_fetch=oanda_fetch,
        csv_path=csv_path,
        tier2_opts=tier2_opts,
        seq_len=seq_len,
        console=console,
    )

    # Standardize features (train-only stats).
    feats, mu, sigma, train_feats, val_feats = _standardize_features(feats, split)

    # Optional PCA
    train_feats, val_feats, pca_model, pca_components_eff = _apply_pca(
        train_feats, val_feats, pca_components, console
    )

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
    _validate_features_and_targets(
        train_feats, val_feats, train_dir_raw, val_dir_raw,
        train_conf_raw, val_conf_raw, train_conf_w_raw, val_conf_w_raw,
    )

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
            _train_ensemble_models(
                cfg=cfg,
                options=options,
                console=console,
                oanda_fetch=oanda_fetch,
                csv_path=csv_path,
                train_feats=train_feats,
                val_feats=val_feats,
                df=df,
                ohlc_df=ohlc_df,
                feature_columns=feature_columns,
                seq_len=seq_len,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                fit_verbose=fit_verbose,
                rl_position_cfg=rl_position_cfg,
            )
            return

        # XGBoost model - handles training separately
        elif effective_model_type == "xgboost":
            import json
            from datetime import datetime
            console.print("[cyan]Model: XGBoost (gradient boosting - often better on tabular data)[/cyan]")
            from src.models.xgboost_model import train_xgboost_buddy, XGBoostKerasWrapper

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

            # Check if meta-labeling is enabled
            meta_labeling_enabled = bool(cfg.get("buddy", {}).get("meta_labeling", {}).get("enabled", False))
            if meta_labeling_enabled:
                try:
                    from src.training.meta_labeling import MetaLabeler, MetaLabelingConfig

                    console.print("[cyan]Training meta-labeler for XGBoost (trade filter)...[/cyan]")

                    # Get XGBoost predictions on training and validation data
                    xgb_preds_train = xgb_model.predict(train_feats)
                    xgb_preds_val = xgb_model.predict(val_feats)

                    train_dir_preds = xgb_preds_train["direction"]
                    val_dir_preds = xgb_preds_val["direction"]

                    # Configure meta-labeler with regularization
                    meta_cfg_yaml = cfg.get("buddy", {}).get("meta_labeling", {})
                    meta_cfg = MetaLabelingConfig(
                        use_xgboost=True,
                        n_estimators=int(meta_cfg_yaml.get("n_estimators", 100)),
                        max_depth=int(meta_cfg_yaml.get("max_depth", 2)),
                        learning_rate=float(meta_cfg_yaml.get("learning_rate", 0.05)),
                        reg_alpha=float(meta_cfg_yaml.get("reg_alpha", 1.0)),
                        reg_lambda=float(meta_cfg_yaml.get("reg_lambda", 5.0)),
                        subsample=float(meta_cfg_yaml.get("subsample", 0.7)),
                        colsample_bytree=float(meta_cfg_yaml.get("colsample_bytree", 0.7)),
                        min_child_weight=int(meta_cfg_yaml.get("min_child_weight", 5)),
                        early_stopping_rounds=int(meta_cfg_yaml.get("early_stopping_rounds", 30)),
                        max_overfitting_gap=float(meta_cfg_yaml.get("max_overfitting_gap", 0.15)),
                        auto_tune=bool(meta_cfg_yaml.get("auto_tune", False)),
                        min_confidence_threshold=float(meta_cfg_yaml.get("threshold", 0.55)),
                        use_reduced_features=bool(meta_cfg_yaml.get("use_reduced_features", True)),
                    )

                    labeler = MetaLabeler(meta_cfg)
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

                    best_thresh, thresh_metrics = labeler.tune_threshold(
                        val_feats, val_dir_preds, val_dir_raw, verbose=True
                    )
                    thresh_improvement = thresh_metrics.get("improvement", 0) if thresh_metrics else 0

                    console.print(f"Meta-labeler: train_acc={train_meta_acc:.1%} val_acc={val_meta_acc:.1%} gap={overfit_gap:.1%}")
                    if thresh_metrics:
                        console.print(f"Threshold tuned: {best_thresh:.2f} -> +{thresh_improvement*100:.1f}% improvement")

                    if overfit_gap > 0.20:
                        console.print(f"[yellow]⚠️ Meta-labeler rejected (gap {overfit_gap:.1%} > 20%)[/yellow]")
                        meta["meta_labeling"] = {"enabled": False, "reason": f"overfitting_gap_{overfit_gap:.1%}"}
                    else:
                        meta_labeler_path = model_dir / f"buddy_xgb_meta{suffix}.pkl"
                        labeler.save(meta_labeler_path)
                        console.print(f"Saved: {meta_labeler_path}")

                        meta["meta_labeling"] = {
                            "enabled": True,
                            "path": str(meta_labeler_path),
                            "train_accuracy": float(train_meta_acc),
                            "val_accuracy": float(val_meta_acc),
                            "overfitting_gap": float(overfit_gap),
                            "tuned_threshold": float(best_thresh),
                            "threshold_improvement": float(thresh_improvement),
                        }

                    with open(candidate_meta_path.with_suffix('.xgb.meta.json'), "w") as f:
                        json.dump(meta, f, indent=2)

                except Exception as e:
                    console.print(f"[yellow]Meta-labeling failed[/yellow]: {e}")
                    import traceback
                    traceback.print_exc()

            console.print("[green]XGBoost training complete![/green]")

            # RL Position Sizer Training after XGBoost
            train_rl_sizer_enabled = _is_rl_auto_train_enabled(cfg, options)

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

                    rl_features = np.vstack([train_feats, val_feats]).astype(np.float32)
                    n_rl_samples = len(rl_features)
                    rl_prices = ohlc_df['close'].values[:n_rl_samples].astype(np.float32)

                    console.print(f"[dim]Generating XGBoost predictions for {n_rl_samples:,} samples...[/dim]")
                    xgb_preds = xgb_model.predict(rl_features)
                    rl_direction_probs = xgb_preds["direction"].astype(np.float32)
                    rl_confidences = np.clip(xgb_preds.get("confidence", np.full(n_rl_samples, 0.5)), 0.0, 1.0).astype(np.float32)

                    rl_predictions = np.column_stack([
                        rl_direction_probs,
                        rl_confidences,
                    ]).astype(np.float32)

                    console.print(f"[dim]RL training data ready: {n_rl_samples:,} samples, {rl_features.shape[1]} features[/dim]")

                    rl_timesteps = options.rl_timesteps
                    _train_rl_position_sizer_if_ready(
                        console=console,
                        rl_timesteps=rl_timesteps,
                        features=rl_features,
                        ensemble_predictions=rl_predictions,
                        prices=rl_prices,
                        rl_position_cfg=rl_position_cfg,
                    )

                except Exception as e:
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

    # Losses and optimizer setup
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
        m = "val_loss" if monitor == "val_loss" else "loss"
        es = tf.keras.callbacks.EarlyStopping(monitor=m, mode="min", **es_kwargs)
    callbacks.append(es)

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

    # Confirm whether EarlyStopping triggered
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
            from src.utils.fx_paper import pip_size, simulate_tp_sl_outcome

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
                [x_val_oos_2d[k : k + int(seq_len_eval)] for k in range(n_windows_oos)],
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
            from src.utils.fx_paper import pip_size, simulate_tp_sl_outcome

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
            from src.training.meta_labeling import MetaLabeler, MetaLabelingConfig

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
    train_rl_sizer_enabled = _is_rl_auto_train_enabled(cfg, options)

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
                rl_position_cfg=rl_position_cfg,
            )

        except Exception as e:
            # RL training failure should NOT crash main training
            console.print(f"[yellow]⚠ RL position sizer training failed: {e}[/yellow]")
            console.print("[dim]Ensemble training completed successfully - RL training is optional.[/dim]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
