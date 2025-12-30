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

import sys
import time
import logging
import argparse
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from utils import setup_logging, load_config

if TYPE_CHECKING:
    from neural_network_integrator_enhanced import NeuralNetworkIntegrator


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
    oanda_fetch: OandaFetchOptions | None = None
    advanced: BuddyTrainingAdvancedOptions | None = None
    feature_curriculum: bool = False
    curriculum_ks: str | None = None
    pca_components: int | None = None
    seq_len: int = 50
    epochs: int = 300
    batch_size: int = 32
    lr: float = 0.001
    patience: int = 10
    seed: int = 42
    run_tag: str | None = None
    all_features: bool = True
    device: str = "auto"  # auto|cpu|gpu
    mixed_precision: bool = False
    steps_per_execution: int = 1
    jit_compile: bool = False
    shuffle_buffer: int | None = None
    prefetch: int | None = None
    cache_val: bool = False
    combined_use_predict: bool = True
    shared_encoder: bool = False
    timing: bool = False
    fit_verbose: int = 1


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


def _tier2_apply_calibration(meta: dict[str, Any], score: float) -> float | None:
    """Map a model score in [0,1] to calibrated P(win) via saved bins."""
    try:
        calib = _tier2_get_calibration_dict(meta)
        if calib is None:
            return None
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
    resp = client.get_candles(
        opts.instrument,
        granularity=str(opts.granularity),
        count=int(opts.candles),
        price=str(opts.price),
    )
    oanda_df = candles_to_ohlcv_df(resp)

    # Helpful "recency" logging so it's obvious we pulled live data now.
    try:
        ts_fmt = "%Y-%m-%dT%H:%M:%SZ"
        now_utc = datetime.now(timezone.utc).strftime(ts_fmt)
        if "time" in oanda_df.columns and len(oanda_df) > 0:
            t = pd.to_datetime(oanda_df["time"], utc=True, errors="coerce")
            if not t.isna().all():
                first_t = t.min().strftime(ts_fmt)
                last_t = t.max().strftime(ts_fmt)
                console.print(
                    f"OANDA fetch @ {now_utc} | candle range {first_t} .. {last_t} | rows={len(oanda_df)}"
                )
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
    console.print(f"Fetched OANDA candles -> {out_path} ({len(oanda_df)} rows)")
    return str(out_path)


def _load_buddy_checkpoint(*, model_path: str, seq_len: int, feature_dim: int):
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
DEFAULT_CONFIG_PATH = "./config.yaml"
DEFAULT_MESSAGE_FORMAT = "Epoch {epoch} completed"
TIMESTAMP_FORMAT = "%H:%M:%S"
TABLE_HEADER_STYLE = "bold magenta"
DEFAULT_CURRICULUM_KS = "32,64,128,0"

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


def _normalize_instrument(s: str) -> str:
    v = (s or "").strip().upper().replace("-", "_").replace("/", "_")
    v = "_".join([p for p in v.split("_") if p])
    if "_" not in v and len(v) == 6:
        v = v[:3] + "_" + v[3:]
    return v


def launch_buddy_repl_from_wizard(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    instrument: str = "USD_JPY",
    granularity: str = "M5",
    candles: int = 2000,
    execute: bool = True,
    force_execute: bool = False,
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


def _buddy_interactive_wizard(*, default_config: str = DEFAULT_CONFIG_PATH) -> None:
    def _ask(prompt: str) -> str:
        try:
            return str(console.input(prompt) or "")
        except EOFError:
            return ""

    console.print("[bold]Buddy interactive[/bold] (press Enter for defaults)")

    cfg_path = (_ask(f"Config path [{default_config}]: ") or "").strip() or default_config
    instrument = _normalize_instrument(_ask("Instrument [USD_JPY]: ") or "USD_JPY")
    granularity = (_ask("Granularity [M5]: ") or "M5").strip() or "M5"

    candles = _parse_int_answer(
        _ask("Candles lookback [2000]: "),
        default=2000,
    )
    if int(candles) < 300:
        candles = 300

    equity = _parse_float_answer(_ask("Equity for sizing (PRACTICE NAV fallback) [10000]: "), default=10_000.0)

    # Risk is used as a fraction internally (0.005 = 0.5%). Ask as percent for humans.
    risk_pct = _parse_float_answer(_ask("Risk per trade (%) [0.5]: "), default=0.5)
    risk_frac = max(0.0, float(risk_pct) / 100.0)

    execute = _parse_bool_answer(_ask("Place PRACTICE orders? (Y/n): "), default=True)
    dry_run = False
    if not execute:
        dry_run = True

    force_execute = False
    if execute:
        force_execute = _parse_bool_answer(
            _ask("Bypass training gate (>=62% dir acc)? (Y/n): "),
            default=True,
        )

    loop = _parse_bool_answer(_ask("Run continuously (loop)? (y/N): "), default=False)
    max_trades = 1
    if loop:
        max_trades = _parse_int_answer(_ask("Max executed trades (0 = unlimited) [1]: "), default=1)
        max_trades = max(0, int(max_trades))

    console.print(
        f"\n[dim]Summary[/dim]: instrument={instrument} granularity={granularity} candles={int(candles)} "
        f"equity={float(equity):.2f} risk={risk_frac*100:.3f}% execute={bool(execute)} force_execute={bool(force_execute)} loop={bool(loop)}"
    )
    if not _parse_bool_answer(_ask("Proceed? (Y/n): "), default=True):
        console.print("[dim]Cancelled.[/dim]")
        return

    # Offer a short mode menu to make intentions explicit.
    console.print(
        "\nChoose mode:\n 1) Single-shot inference: run one prediction and exit\n"
        " 2) Loop trading: run continuously and execute trades\n"
        " 3) Launch Buddy REPL: interactive trading commands (buy/sell/close/etc.)\n"
        " 4) Train Buddy: run training workflow"
    )
    choice = (_ask("Mode [1]: ") or "1").strip() or "1"
    try:
        mode = int(choice)
    except Exception:
        mode = 1

    if mode == 3:
        # Launch the interactive REPL that exposes Buddy's trade commands.
        launch_buddy_repl_from_wizard(
            cfg_path,
            checkpoint_path=None,
            instrument=instrument,
            granularity=granularity,
            candles=int(candles),
            execute=bool(execute),
            force_execute=bool(force_execute),
            all_features=False,
            verbose=False,
            assistant_name="Buddy",
        )
        return

    if mode == 4:
        # Kick off training flow
        train_buddy(cfg_path)
        return

    # Mode 2 = loop, else single-shot
    if mode == 2 or loop:
        buddy_loop(
            cfg_path,
            instrument=instrument,
            granularity=granularity,
            candles=int(candles),
            execute=True,
            force_execute=bool(force_execute),
            equity=float(equity),
            risk_per_trade_pct=float(risk_frac),
            all_features=False,
            verbose=False,
            max_trades=int(max_trades),
        )
        return

    buddy(
        cfg_path,
        instrument=instrument,
        granularity=granularity,
        candles=int(candles),
        execute=True,
        force_execute=bool(force_execute),
        equity=float(equity),
        risk_per_trade_pct=float(risk_frac),
        all_features=False,
        verbose=False,
    )


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
        "neural_network_integrator_enhanced",
        "ml_engine_enhanced",
        "memory_manager_enhanced",
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


def _train_buddy_impl(
    config_path: str,
    csv_path: str | None = None,
    *,
    options: BuddyTrainingOptions,
) -> None:  # noqa: C901, PLR0915, WPS231, CCR001  # NOSONAR
    """Train Buddy (TensorFlow-only) from USDJPY historical data."""
    cfg = load_config(config_path)

    oanda_fetch = options.oanda_fetch
    advanced = options.advanced
    feature_curriculum = options.feature_curriculum
    curriculum_ks = options.curriculum_ks
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
    mixed_precision = options.mixed_precision
    steps_per_execution = options.steps_per_execution
    jit_compile = options.jit_compile
    shuffle_buffer = options.shuffle_buffer
    prefetch = options.prefetch
    cache_val = options.cache_val
    combined_use_predict = options.combined_use_predict
    shared_encoder = options.shared_encoder
    timing = options.timing
    fit_verbose = options.fit_verbose

    dev = str(device or "auto").strip().lower()
    force_cpu = dev == "cpu"
    _configure_tf_metal(verbose=False, force_cpu=force_cpu)

    try:
        from tracing_setup import setup_tracing as _setup_tracing

        _setup_tracing()
    except Exception:
        # tracing should never break runtime
        pass
    import json
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from feature_engineering import FeatureEngineering

    buddy_cfg = (cfg.get("buddy") or {}) if isinstance(cfg, dict) else {}

    # Optional mixed precision (behind a flag). Some TF-Metal builds benefit, some don't.
    mp_enabled = bool(mixed_precision)
    if mp_enabled:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            console.print("Mixed precision enabled: mixed_float16")
        except Exception as e:
            console.print(f"[yellow]Mixed precision enable failed; continuing in float32[/yellow]: {e}")
            mp_enabled = False

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
    tier2_stop_loss_pips_cfg = float(buddy_cfg.get("stop_loss_pips", 20.0))
    tier2_take_profit_pips_cfg = float(buddy_cfg.get("take_profit_pips", 60.0))
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

    if oanda_fetch is not None:
        # Fetch fresh candles from OANDA PRACTICE and train from that dataset.
        csv_path = _oanda_fetch_to_csv(oanda_fetch)
    elif csv_path is None:
        # Default to repo-local clean USDJPY M5 data.
        csv_path = str(Path("market_data") / "oanda_USD_JPY_M5.csv")

    # If the default local CSV doesn't exist, try the most recent live fetch.
    csv_p = Path(str(csv_path))
    if not csv_p.exists():
        try:
            md = Path("market_data")
            candidates = sorted(md.glob("oanda_USD_JPY_M5_live_*.csv"), key=lambda p: p.stat().st_mtime)
            if candidates:
                csv_p = candidates[-1]
                csv_path = str(csv_p)
                console.print(f"Default CSV missing; using latest live dataset: {csv_path}")
        except Exception:
            pass

    if not Path(str(csv_path)).exists():
        raise FileNotFoundError(
            f"Training CSV not found: {csv_path}. Use --oanda-live to fetch, or pass --csv <path>."
        )

    df = pd.read_csv(csv_path)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time")

    # Ensure expected OHLCV columns exist.
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Optional data-quality filters.
    if min_volume is not None:
        try:
            before = int(len(df))
            df = df.loc[df["volume"].astype(float) >= float(min_volume)].copy()
            removed = before - int(len(df))
            if removed > 0:
                console.print(f"Filtered low-volume candles: removed {removed} rows (min_volume={min_volume})")
        except Exception as e:
            console.print(f"[yellow]Min-volume filter skipped[/yellow]: {e}")

    if spread_filter:
        # OANDA candles may include bid/ask close columns when price includes B/A.
        if "bid_close" in df.columns and "ask_close" in df.columns:
            try:
                before = int(len(df))
                spread = (df["ask_close"].astype(float) - df["bid_close"].astype(float)).abs()
                spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
                if not spread.empty:
                    pctl_thr = float(np.quantile(spread.to_numpy(), float(spread_pctl)))
                    med = float(np.median(spread.to_numpy()))
                    thr = max(pctl_thr, float(spread_mult) * med) if med > 0 else pctl_thr
                    keep_mask = (df["ask_close"].astype(float) - df["bid_close"].astype(float)).abs() <= thr
                    df = df.loc[keep_mask].copy()
                    removed = before - int(len(df))
                    if removed > 0:
                        console.print(
                            f"Filtered wide-spread candles: removed {removed} rows (pctl={spread_pctl}, mult={spread_mult}, thr={thr:.10g})"
                        )
            except Exception as e:
                console.print(f"[yellow]Spread filter skipped[/yellow]: {e}")
        else:
            console.print("[yellow]Spread filter requested but bid/ask columns missing; skipping[/yellow]")

    # Training-time noise reduction:
    # - If we are running full feature engineering, let FeatureEngineering apply it (so it happens exactly once).
    # - If feature engineering is disabled, apply it here so Buddy still trains on de-noised candles.
    if train_smoothing:
        extra = f" + median(close,w={median_window})" if median_window else ""
        console.print(f"Training noise reduction enabled: resample->M5{extra} + EMA(14) close")

    # Feature engineering (numeric only).
    if all_features:
        fe = FeatureEngineering(cfg.get("feature_engineering", {}))
        df = fe.create_features(
            df,
            include_all=True,
            apply_candle_smoothing=train_smoothing,
            median_window=median_window,
        )
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
                    f"Top-feature selection enabled: kept {len(keep)} / {int(len(ranked_cols))} ranked numeric features"
                )
        except Exception as e:
            console.print(f"[yellow]Feature ranking skipped[/yellow]: {e}")
            ranked_cols = None

    feats = numeric_df.to_numpy(dtype=np.float32)
    feature_columns = list(numeric_df.columns)

    if feats.shape[1] < 6:
        console.print(f"[yellow]Warning[/yellow]: only {feats.shape[1]} numeric features detected.")
    else:
        console.print(f"Features: {feats.shape[1]} numeric columns")

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
        f"Direction label positive rate (train/val): {train_dir_pos_rate*100:.2f}% / {val_dir_pos_rate*100:.2f}%"
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

    try:
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
        for gi in range(int(start_gi), int(max_signal) + 1, int(label_stride)):
            spread_pips = _spread_pips_at_entry(int(gi))
            res_long = simulate_tp_sl_outcome(
                ohlc_df,
                instrument=str(instrument_guess),
                signal_index=int(gi),
                direction="long",
                stop_loss_pips=float(tier2_stop_loss_pips),
                take_profit_pips=float(tier2_take_profit_pips),
                spread_pips=float(spread_pips),
                max_horizon_candles=int(tier2_horizon_candles),
                tie_break=str(tier2_tie_break),
            )
            res_short = simulate_tp_sl_outcome(
                ohlc_df,
                instrument=str(instrument_guess),
                signal_index=int(gi),
                direction="short",
                stop_loss_pips=float(tier2_stop_loss_pips),
                take_profit_pips=float(tier2_take_profit_pips),
                spread_pips=float(spread_pips),
                max_horizon_candles=int(tier2_horizon_candles),
                tie_break=str(tier2_tie_break),
            )

            win_long = int(res_long.to_label())
            win_short = int(res_short.to_label())
            conf_y_all[int(gi)] = float(win_long) if float(direction_y_all[int(gi)]) >= 0.5 else float(win_short)
            conf_w_all[int(gi)] = 1.0

        console.print(
            "Confidence targets: Tier-2 TP/SL v1 "
            f"(instrument={str(instrument_guess)}, SL={float(tier2_stop_loss_pips):g}, TP={float(tier2_take_profit_pips):g}, "
            f"horizon={int(tier2_horizon_candles)}, label_stride={int(label_stride)})"
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
        f"Confidence target mean (train/val): {train_conf_target_mean*100:.2f}% / {val_conf_target_mean*100:.2f}% "
        f"(weighted; Tier-2 TP/SL)"
    )

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

    feature_dim = int(train_feats.shape[1])
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

        if shared_encoder:
            console.print("Model: shared encoder (fast path)")
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
            model = _build_buddy_model(
                feature_dim=int(feature_dim),
                seq_len=seq_len,
                head_hidden=int(head_hidden),
                head_layers=int(head_layers),
                head_dropout=float(head_dropout),
                dense_hidden=int(dense_hidden),
                dense_dropout=float(dense_dropout),
            )

    # Losses:
    # - direction: binary classification (sigmoid)
    # - confidence: TP-before-SL (Tier-2) probability label (sigmoid)
    bce_dir = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    bce_conf = tf.keras.losses.BinaryCrossentropy(from_logits=False)
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
    _compile_with_optional_args(
        model,
        optimizer=optimizer,
        loss={"direction": bce_dir, "confidence": bce_conf},
        metrics={
            "direction": [tf.keras.metrics.BinaryAccuracy(name="accuracy")],
            "confidence": [tf.keras.metrics.BinaryAccuracy(name="accuracy")],
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
        ):
            super().__init__()
            self._val_ds_fit = val_ds_fit
            self._validation_steps = int(validation_steps)
            self._w_dir = float(w_dir)
            self._w_conf = float(w_conf)
            self._use_predict = bool(use_predict)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs if isinstance(logs, dict) else {}

            if self._use_predict:
                try:
                    pred = self.model.predict(self._val_ds_fit, steps=self._validation_steps, verbose=0)
                    conf = np.asarray(pred["confidence"]).reshape(-1)
                    conf = np.clip(conf, 0.0, 1.0)
                    avg_conf = float(np.mean(conf))
                except Exception:
                    return
            else:
                # Fast path: avoid an extra predict() pass by using a proxy metric.
                # This is not identical to mean predicted confidence, but is sufficient
                # for a throughput-focused early-stopping signal.
                try:
                    avg_conf = float(logs.get("val_confidence_accuracy", 0.0) or 0.0)
                except Exception:
                    avg_conf = 0.0

            logs["val_avg_confidence"] = avg_conf
            val_dir_acc = float(logs.get("val_direction_accuracy", 0.0) or 0.0)
            logs["val_combined_score"] = (self._w_dir * val_dir_acc) + (self._w_conf * avg_conf)

    monitor = str(es_monitor or "direction").strip().lower()
    if monitor not in {"direction", "combined", "val_loss", "loss"}:
        raise ValueError("es_monitor must be one of: direction, combined, val_loss, loss")

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

    # Use the repeated validation dataset with explicit steps to avoid Keras
    # "input ran out of data" warnings when dataset cardinality is unknown.
    pred = model.predict(val_ds_fit, steps=int(validation_steps), verbose=0)
    dir_pred = (np.asarray(pred["direction"]).reshape(-1) >= 0.5).astype(np.float32)
    # Pull true labels from exactly the same number of validation batches.
    val_ds_eval = val_ds.take(int(validation_steps))
    try:
        dir_true_batches = []
        conf_true_batches = []
        for batch in val_ds_eval:
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                _, yb, _ = batch
            else:
                _, yb = batch
            dir_true_batches.append(np.asarray(yb["direction"]).reshape(-1))
            conf_true_batches.append(np.asarray(yb["confidence"]).reshape(-1))
        dir_true = np.concatenate(dir_true_batches, axis=0).astype(np.float32)
        conf_true = np.concatenate(conf_true_batches, axis=0).astype(np.float32)
    except Exception:
        dir_true = None
        conf_true = None

    val_acc = float((dir_pred == dir_true).mean()) if dir_true is not None else float("nan")
    conf_preds = np.asarray(pred["confidence"]).reshape(-1)
    conf_preds = np.clip(conf_preds, 0.0, 1.0)
    avg_conf = float(np.mean(conf_preds))

    try:
        val_conf_mae = float(np.mean(np.abs(conf_preds - conf_true.reshape(-1)))) if conf_true is not None else None
    except Exception:
        val_conf_mae = None

    # Confidence distribution (helps set realistic live thresholds).
    try:
        val_conf_p80 = float(np.quantile(conf_preds, 0.80))
        val_conf_p90 = float(np.quantile(conf_preds, 0.90))
    except Exception:
        val_conf_p80 = None
        val_conf_p90 = None

    combined_score = (float(combined_w_dir) * val_acc) + (float(combined_w_conf) * avg_conf)

    console.print(f"Validation directional accuracy: [bold]{val_acc*100:.2f}%[/bold]")
    console.print(f"Validation average confidence: [bold]{avg_conf*100:.2f}%[/bold]")
    console.print(f"Validation target mean confidence: [bold]{val_conf_target_mean*100:.2f}%[/bold]")
    console.print(f"Validation combined score: [bold]{combined_score*100:.2f}%[/bold]")

    model_dir = Path("trained_data") / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_tag}" if run_tag else ""
    model_path = model_dir / f"buddy_tf{suffix}.keras"
    meta_path = model_dir / f"buddy_tf{suffix}.meta.json"

    # Save model and metadata for inference/trading.
    model.save(model_path)
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
    try:
        if tier2_calibrate:
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
                w = feats[int(gi) - int(seq_len) + 1 : int(gi) + 1]
                if w.shape[0] != int(seq_len):
                    continue
                batch_x.append(np.asarray(w, dtype=np.float32))
                batch_idx.append(int(gi))
                if len(batch_x) >= batch_max:
                    _flush_batch()
            _flush_batch()

            if len(scores) >= 50:
                s = np.asarray(scores, dtype=np.float64)
                y = np.asarray(wins, dtype=np.float64)

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
                            "bins": bins_out,
                        }
                        console.print(
                            f"Tier-2 calibration: n={int(tier2_calibration['n'])} win_rate={float(tier2_calibration['win_rate'])*100:.2f}% bins={len(bins_out)}"
                        )
            else:
                console.print(
                    f"[yellow]Tier-2 calibration skipped[/yellow]: insufficient labeled samples (n={len(scores)})"
                )
    except Exception as e:
        console.print(f"[yellow]Tier-2 calibration failed[/yellow]: {e}")
        tier2_calibration = None

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
        "tier2": (
            {
                "enabled": bool(tier2_calibrate and tier2_calibration is not None),
                "calibration": tier2_calibration,
            }
            if tier2_calibration is not None
            else {"enabled": False, "calibration": None}
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    console.print(f"Saved: {model_path}")
    console.print(f"Saved: {meta_path}")


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
    mid = (bid + ask) / 2.0
    half_spread = max(0.0, ask - mid)
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


def fx_paper_trade(
    config_path: str,
    instrument: str,
    granularity: str = "M5",
    candles: int = 300,
    execute: bool = False,
    equity: float = 10_000.0,
    risk_per_trade_pct: float = 0.005,
) -> None:
    """Paper trade forex on OANDA PRACTICE using a simple setup rule.

    This is intentionally conservative and defaults to DRY-RUN (no order placed).
    """
    cfg = load_config(config_path)

    from oanda_practice import OandaPracticeClient
    import fx_guardrails as fxg

    policy = _fx_enforce_fx_policy(cfg, instrument=instrument, granularity=granularity)
    if policy is None:
        return

    client = OandaPracticeClient.from_env()
    state = fxg.load_state(cfg, policy)

    pnl, _ = _fx_refresh_fx_state(cfg, policy, state, client)
    if _fx_maybe_force_flat(policy, state, client, execute=execute):
        return

    if not _fx_require_account_metrics(pnl):
        return

    if not _fx_gate_fx_entry(policy, state, client):
        return

    df = _fx_load_fx_df(client, instrument=instrument, granularity=granularity, candles=candles)
    ok_spread, spread_pips, slippage_pips, slippage_price = _fx_spread_and_slippage(policy, df, instrument=instrument)
    if not ok_spread:
        return

    signal, last_close, atr_value = _fx_get_signal_context(df)
    if signal == "hold":
        console.print(f"FX setup: HOLD  {instrument} {granularity}  close={last_close:.5f}  ATR14={atr_value:.5f}")
        return

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
        return

    if _fx_apply_daily_stops(cfg, policy, state, pnl, band=band):
        return

    units, stop_price, tp_price, _, _ = _fx_build_order_units_and_prices(
        policy,
        rules,
        instrument=instrument,
        signal=signal,
        price=last_close,
        atr_value=atr_value,
        slippage_price=float(slippage_price),
    )

    console.print(
        f"FX setup: {signal.upper()}  {instrument} {granularity}  close={last_close:.5f}  units={units}  SL={stop_price:.5f}  TP={tp_price:.5f}"
        f"  spread={spread_pips:.2f}p  slip={slippage_pips:.2f}p  conf={confidence:.2f}  band={band}"
    )

    if not execute:
        console.print("[dim]Dry-run (no order). Pass --execute to place a PRACTICE order.[/dim]")
        return

    if policy.require_confirmation and not _fx_confirm("Confirm PLACE ORDER? (y/N): "):
        console.print("[dim]Order cancelled.[/dim]")
        return

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

    price_bound = _fx_execution_guard_price_bound(policy, client, instrument=instrument, units=units_final)
    if price_bound is None:
        return

    result = client.create_market_order(
        instrument=instrument,
        units=units_final,
        stop_loss_price=stop_price,
        take_profit_price=tp_price,
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
            tro = tx.get("tradesOpened") or tx.get("tradesOpened")
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
                _schedule_auto_close(client, instrument, delay_s=m * 60.0, verbose=bool(verbose))
    except Exception:
        pass


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
        _ = openai_integration.report_config_changes(current_config, result)
        console.print(
            "[bold green]Configuration updated successfully and written to config.yaml[/bold green]"
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


def buddy(
    config_path: str,
    *,
    checkpoint_path: str | None = None,
    instrument: str = "USD_JPY",
    granularity: str = "M5",
    candles: int = 300,
    execute: bool = False,  # Live trading must be explicitly enabled
    force_execute: bool = False,
    force_units: int | None = None,
    max_hold_minutes: float | None = None,
    equity: float | None = None,
    risk_per_trade_pct: float | None = None,
    all_features: bool = True,
    verbose: bool = False,
) -> None:
    """Buddy: run one TF-only inference on fresh OANDA candles; optionally place a trade."""
    _configure_predict_output(verbose)

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

    # Load model + preprocessing metadata.
    meta_path = Path("trained_data") / "models" / BUDDY_META_FILENAME
    if checkpoint_path:
        ck_meta = _meta_path_for_checkpoint(str(checkpoint_path))
        if ck_meta and ck_meta.exists():
            meta_path = ck_meta
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

    if verbose:
        conf_msg = f"model_conf={_fmt_prob(p_conf)}"
        if tier1_conf is not None:
            conf_msg += f" tier1_conf={_fmt_prob(gate_conf)}"
        if tier2_p_win is not None:
            conf_msg += f" tier2_p_win={_fmt_prob(float(tier2_p_win))}"
        console.print(f"Buddy prediction: direction={_fmt_prob(p_dir)} ({side}), {conf_msg}")
        if tier1_reasons:
            console.print(f"[dim]Tier-1 confidence reasons[/dim]: {', '.join(tier1_reasons)}")
        if p_dir <= 1e-12 or p_dir >= 1.0 - 1e-12 or p_conf <= 1e-12 or p_conf >= 1.0 - 1e-12:
            dlog = _dense_logit("direction")
            clog = _dense_logit("confidence")
            if dlog is not None or clog is not None:
                console.print(
                    f"[dim]Inference diag[/dim]: direction_logit={dlog if dlog is not None else 'n/a'} "
                    f"confidence_logit={clog if clog is not None else 'n/a'} (float32 sigmoid saturation can clamp to 0/1)"
                )
                try:
                    if dlog is not None:
                        console.print(f"[dim]Inference diag[/dim]: direction_sigmoid64={_stable_sigmoid_from_logit(dlog):.3e}")
                    if clog is not None:
                        console.print(f"[dim]Inference diag[/dim]: confidence_sigmoid64={_stable_sigmoid_from_logit(clog):.3e}")
                except Exception:
                    pass
    else:
        if tier2_p_win is not None:
            console.print(
                f"Buddy prediction: direction={p_dir:.3f} ({side}), model_conf={p_conf:.3f}, tier2_p_win={float(tier2_p_win):.3f}"
            )
        elif tier1_conf is not None:
            console.print(
                f"Buddy prediction: direction={p_dir:.3f} ({side}), model_conf={p_conf:.3f}, tier1_conf={gate_conf:.3f}"
            )
        else:
            console.print(f"Buddy prediction: direction={p_dir:.3f} ({side}), confidence={p_conf:.3f}")

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
    # Tier-2 sizing: fractional Kelly for fixed RR, capped by config.
    risk_eff = None
    if tier2_p_win is not None:
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
    candles: int = 300,
    execute: bool = False,
    force_execute: bool = False,
    force_units: int | None = None,
    equity: float | None = None,
    risk_per_trade_pct: float | None = None,
    all_features: bool = True,
    verbose: bool = False,
    max_trades: int = 1,
) -> None:
    """Buddy loop: keep TF+model warm; trade once per new candle (practice only)."""
    _configure_predict_output(verbose)

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
            eta = datetime.fromtimestamp(next_boundary, tz=timezone.utc).isoformat().replace("+00:00", "Z")
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
        return ts.isoformat().replace("+00:00", "Z")

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


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="ML Engine Trading Bot CLI")
    parser.add_argument(
        "command",
        nargs="?",
        default="buddy",
        choices=[
            "train-buddy",
            "buddy",
            "Buddy",
        ],
        help="Command to execute",
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
        default=50,
        help="Sequence length for Buddy training (default 50)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Epochs for Buddy training (default 300)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for Buddy training (default 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Buddy training (default 0.001)",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="For train-buddy: initialize weights from trained_data/models/buddy_tf.keras before training",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="For train-buddy: path to a .keras checkpoint to warm-start from (overrides --warm-start)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="EarlyStopping patience for Buddy training (applies to --es-monitor) (default 10)",
    )
    parser.add_argument(
        "--es-monitor",
        default="direction",
        choices=["direction", "combined", "val_loss", "loss"],
        help="EarlyStopping monitor for train-buddy: direction|combined|val_loss|loss (default direction)",
    )
    parser.add_argument(
        "--combined-w-dir",
        type=float,
        default=0.7,
        help="Weight for direction accuracy in combined early-stopping objective (default 0.7)",
    )
    parser.add_argument(
        "--combined-w-conf",
        type=float,
        default=0.3,
        help="Weight for avg confidence in combined early-stopping objective (default 0.3)",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=None,
        help="For train-buddy: keep only top-K most correlated numeric features (disables warm-start)",
    )
    parser.add_argument(
        "--feature-curriculum",
        action="store_true",
        help="For train-buddy: start with top-K features and automatically unlock more as accuracy improves (cannot be used with --top-features)",
    )
    parser.add_argument(
        "--curriculum-ks",
        default=DEFAULT_CURRICULUM_KS,
        help=f"For train-buddy --feature-curriculum: comma-separated K schedule (0 = all). Default: {DEFAULT_CURRICULUM_KS}",
    )
    parser.add_argument(
        "--median-window",
        type=int,
        default=None,
        help="For train-buddy: apply rolling median to close before EMA smoothing (e.g. 3,5)",
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
        default="USD_JPY",
        help="OANDA instrument e.g. USD_JPY,EUR_USD,GBP_USD (used by buddy and --oanda-live training)",
    )
    parser.add_argument(
        "--granularity",
        "-g",
        default="M5",
        help="OANDA candle granularity, e.g. M5, M15, H1 (used by buddy and --oanda-live training)",
    )
    parser.add_argument(
        "--candles",
        "-n",
        type=int,
        default=300,
        help="How many candles to fetch (used by buddy and --oanda-live training)",
    )

    parser.add_argument(
        "--oanda-live",
        action="store_true",
        help="For train-buddy: fetch fresh candles from OANDA PRACTICE (ignores --csv)",
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
        action="store_true",
        help="Use all engineered features (technical indicators, statistical, time, lag, rolling) for training",
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
        action="store_true",
        help="Enable mixed precision (mixed_float16) for Buddy training (experimental on TF-Metal)",
    )
    parser.add_argument(
        "--steps-per-execution",
        type=int,
        default=1,
        help="Keras steps_per_execution to reduce per-step overhead (default 1)",
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
        help="For --es-monitor combined: use an extra predict() pass per epoch to compute mean confidence (default: enabled; disable for speed)",
    )
    parser.add_argument(
        "--shared-encoder",
        action="store_true",
        help="Use a single shared LSTM encoder instead of 5 parallel heads (much faster, may change accuracy; incompatible with warm-start)",
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
        default=1,
        help="Keras fit verbosity for train-buddy: 0=silent, 1=progress bar, 2=one line per epoch (best for live monitoring)",
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
    # ... add more CLI arguments as needed ...

    # If invoked as `buddy` with no extra args, run the interactive wizard.
    # (The installed `buddy` launcher calls: python main.py buddy ...)
    if len(sys.argv) == 2 and sys.argv[1] in {"buddy", "Buddy"} and sys.stdin.isatty():
        if sys.argv[1] == "Buddy":
            console.print("[yellow]Deprecation: please use lowercase 'buddy' instead of 'Buddy'[/yellow]")
        try:
            _buddy_interactive_wizard(default_config="./config.yaml")
            return
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            return

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Normalize command name and warn on deprecated casing.
    if getattr(args, "command", None) == "Buddy":
        console.print("[yellow]Deprecation: use lowercase 'buddy' instead of 'Buddy'[/yellow]")
        args.command = "buddy"

    # If user requested direct REPL, handle it immediately (before heavy dispatch).
    if getattr(args, "command", None) == "buddy" and bool(getattr(args, "repl", False)):
        launch_buddy_repl_from_wizard(
            args.config,
            checkpoint_path=getattr(args, "model_path", None),
            instrument=str(getattr(args, "instrument", "USD_JPY")),
            granularity=str(getattr(args, "granularity", "M5")),
            candles=int(getattr(args, "candles", 300)),
            execute=bool(getattr(args, "execute", True)),
            force_execute=bool(getattr(args, "force_execute", True)),
            all_features=bool(getattr(args, "all_features", False)),
            verbose=bool(getattr(args, "verbose", False)),
            assistant_name="Buddy",
        )
        return

    command_map = {
        "train-buddy": train_buddy,
        "buddy": buddy,
        "Buddy": buddy,
    }

    try:
        if args.command == "train-buddy":
            # If user didn't override --candles, use a larger default for training fetch.
            train_candles = int(args.candles)
            if bool(getattr(args, "oanda_live", False)) and train_candles == 300:
                train_candles = 5000

            oanda_fetch = None
            if bool(getattr(args, "oanda_live", False)):
                oanda_fetch = OandaFetchOptions(
                    instrument=str(args.instrument),
                    granularity=str(args.granularity),
                    candles=int(train_candles),
                    price=str(getattr(args, "oanda_price", "MBA")),
                    save_csv=getattr(args, "oanda_save_csv", None),
                )

            init_from = getattr(args, "init_from", None)
            if not init_from and bool(getattr(args, "warm_start", False)):
                init_from = str(Path("trained_data") / "models" / "buddy_tf.keras")

            advanced = BuddyTrainingAdvancedOptions(
                init_from=init_from,
                es_monitor=str(getattr(args, "es_monitor", "direction")),
                combined_w_dir=float(getattr(args, "combined_w_dir", 0.7)),
                combined_w_conf=float(getattr(args, "combined_w_conf", 0.3)),
                train_smoothing=not bool(getattr(args, "no_train_smoothing", False)),
                top_features=(
                    int(getattr(args, "top_features", 0))
                    if getattr(args, "top_features", None) is not None
                    else None
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
            )

            command_map[args.command](
                args.config,
                None if bool(getattr(args, "oanda_live", False)) else args.csv,
                oanda_fetch=oanda_fetch,
                advanced=advanced,
                feature_curriculum=bool(getattr(args, "feature_curriculum", False)),
                curriculum_ks=str(getattr(args, "curriculum_ks", DEFAULT_CURRICULUM_KS)),
                pca_components=args.pca_components,
                seq_len=args.seq_len,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                patience=int(args.patience),
                seed=int(args.seed),
                run_tag=args.run_tag,
                all_features=True,
                device=str(getattr(args, "device", "auto")),
                mixed_precision=bool(getattr(args, "mixed_precision", False)),
                steps_per_execution=int(getattr(args, "steps_per_execution", 1)),
                jit_compile=bool(getattr(args, "jit_compile", False)),
                shuffle_buffer=getattr(args, "shuffle_buffer", None),
                prefetch=getattr(args, "prefetch", None),
                cache_val=bool(getattr(args, "cache_val", False)),
                combined_use_predict=bool(getattr(args, "combined_use_predict", True)),
                shared_encoder=bool(getattr(args, "shared_encoder", False)),
                timing=bool(getattr(args, "timing", False)),
                fit_verbose=int(getattr(args, "fit_verbose", 1)),
            )
        elif args.command in {"buddy", "Buddy"}:
            should_execute = bool(args.execute) and (not args.dry_run)
            if bool(getattr(args, "loop", False)):
                buddy_loop(
                    args.config,
                    checkpoint_path=args.model_path,
                    instrument=args.instrument,
                    granularity=args.granularity,
                    candles=args.candles,
                    execute=should_execute,
                    force_execute=bool(getattr(args, "force_execute", False)),
                    force_units=int(getattr(args, "force_units", None)) if getattr(args, "force_units", None) is not None else (
                        int(round(float(getattr(args, "force_margin", 0.0)) / 0.05)) if getattr(args, "force_margin", None) is not None else None
                    ),
                    max_hold_minutes=float(getattr(args, "max_hold_minutes", None)),
                    equity=float(getattr(args, "equity", 10_000.0)),
                    risk_per_trade_pct=float(getattr(args, "risk", 0.005)),
                    all_features=getattr(args, "all_features", False),
                    verbose=args.verbose,
                    max_trades=int(getattr(args, "max_trades", 1)),
                )
            else:
                command_map[args.command](
                    args.config,
                    checkpoint_path=args.model_path,
                    instrument=args.instrument,
                    granularity=args.granularity,
                    candles=args.candles,
                    execute=should_execute,
                    force_execute=bool(getattr(args, "force_execute", False)),
                    force_units=int(getattr(args, "force_units", None)) if getattr(args, "force_units", None) is not None else (
                        int(round(float(getattr(args, "force_margin", 0.0)) / 0.05)) if getattr(args, "force_margin", None) is not None else None
                    ),
                    max_hold_minutes=float(getattr(args, "max_hold_minutes", None)),
                    equity=float(getattr(args, "equity", 10_000.0)),
                    risk_per_trade_pct=float(getattr(args, "risk", 0.005)),
                    all_features=getattr(args, "all_features", False),
                    verbose=args.verbose,
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
