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
from typing import Dict, Any, TYPE_CHECKING

from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from utils import setup_logging, load_config

if TYPE_CHECKING:
    from neural_network_integrator_enhanced import NeuralNetworkIntegrator

# Constants
DEFAULT_MESSAGE_FORMAT = "Epoch {epoch} completed"
TIMESTAMP_FORMAT = "%H:%M:%S"
TABLE_HEADER_STYLE = "bold magenta"

# Initialize console and logging
console = Console()
logger = setup_logging(log_file="cli.log")


def _configure_predict_output(verbose: bool) -> None:
    """Reduce log/warning noise for interactive predict runs."""
    if verbose:
        return

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


def _configure_tf_metal() -> None:
    """Enable TensorFlow Metal GPU on Apple Silicon when available."""
    import os

    # Reduce TF log noise.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        console.print("[yellow]TensorFlow GPU not detected[/yellow] (CPU mode).")
        return

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        console.print(f"[green]TensorFlow GPU enabled[/green]: {gpus}")
    except Exception as e:
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
    """Create the Buddy model: 5 parallel LSTM heads + shared dense + 2 sigmoids."""
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

    direction = tf.keras.layers.Dense(1, activation="sigmoid", name="direction")(x)
    confidence = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence")(x)

    return tf.keras.Model(inputs=inp, outputs={"direction": direction, "confidence": confidence}, name="buddy_model")


def train_buddy(
    config_path: str,
    csv_path: str | None = None,
    *,
    pca_components: int | None = None,
    seq_len: int = 50,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.001,
    all_features: bool = True,
) -> None:
    """Train Buddy (TensorFlow-only) from USDJPY historical data."""
    _configure_tf_metal()
    cfg = load_config(config_path)

    import json
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from feature_engineering import FeatureEngineering

    if csv_path is None:
        # Default to repo-local clean USDJPY M5 data.
        csv_path = str(Path("market_data") / "oanda_USD_JPY_M5.csv")

    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.sort_values("time")

    # Ensure expected OHLCV columns exist.
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Feature engineering (numeric only).
    if all_features:
        fe = FeatureEngineering(cfg.get("feature_engineering", {}))
        df = fe.create_features(df, include_all=True)

    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        raise ValueError("No numeric features found after preprocessing")

    # Drop rows with NaNs created by rolling indicators.
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    closes = df.loc[numeric_df.index, "close"].to_numpy(dtype=np.float32)
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

    # Confidence target: normalized absolute next-return magnitude.
    ret = (next_close - cur_close) / np.maximum(cur_close, 1e-8)
    abs_ret = np.abs(ret).astype(np.float32)

    # Align features to targets (targets are for t->t+1, so drop last feature row).
    feats = feats[:-1]

    n = feats.shape[0]
    if n <= seq_len + 10:
        raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len}")

    # Time-ordered split: last 20% as holdout.
    split = int(n * 0.8)
    train_feats_raw = feats[:split]
    val_feats_raw = feats[split:]
    train_dir_raw = direction_y_all[:split]
    val_dir_raw = direction_y_all[split:]
    train_abs_ret = abs_ret[:split]

    # Standardize features (train-only stats).
    mu = train_feats_raw.mean(axis=0, keepdims=True)
    sigma = train_feats_raw.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    train_feats = (train_feats_raw - mu) / sigma
    val_feats = (val_feats_raw - mu) / sigma

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
        train_feats = pca_model.transform(train_feats)
        val_feats = pca_model.transform(val_feats)
        console.print(f"PCA enabled: {train_feats_raw.shape[1]} -> {train_feats.shape[1]}")

    # Confidence label scale from train only.
    scale = float(np.quantile(train_abs_ret, 0.95))
    if scale <= 0:
        scale = float(train_abs_ret.mean() + 1e-6)

    conf_y_all = np.clip(abs_ret / scale, 0.0, 1.0).astype(np.float32)
    train_conf_raw = conf_y_all[:split]
    val_conf_raw = conf_y_all[split:]

    def make_sequences(x2d: np.ndarray, y_dir: np.ndarray, y_conf: np.ndarray):
        xs: list[np.ndarray] = []
        ys_dir: list[float] = []
        ys_conf: list[float] = []
        for end in range(seq_len, len(x2d)):
            start = end - seq_len
            xs.append(x2d[start:end])
            ys_dir.append(float(y_dir[end]))
            ys_conf.append(float(y_conf[end]))
        x_out = np.stack(xs, axis=0).astype(np.float32)
        y_dir_out = np.asarray(ys_dir, dtype=np.float32).reshape(-1, 1)
        y_conf_out = np.asarray(ys_conf, dtype=np.float32).reshape(-1, 1)
        return x_out, y_dir_out, y_conf_out

    x_train, y_dir_train, y_conf_train = make_sequences(train_feats, train_dir_raw, train_conf_raw)
    x_val, y_dir_val, y_conf_val = make_sequences(val_feats, val_dir_raw, val_conf_raw)

    model = _build_buddy_model(
        feature_dim=x_train.shape[-1],
        seq_len=seq_len,
        head_hidden=int(cfg.get("buddy", {}).get("head_hidden", 64)),
        head_layers=int(cfg.get("buddy", {}).get("head_layers", 2)),
        head_dropout=float(cfg.get("buddy", {}).get("head_dropout", 0.1)),
        dense_hidden=int(cfg.get("buddy", {}).get("dense_hidden", 128)),
        dense_dropout=float(cfg.get("buddy", {}).get("dense_dropout", 0.2)),
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(lr)),
        loss={"direction": "binary_crossentropy", "confidence": "binary_crossentropy"},
        metrics={"direction": [tf.keras.metrics.BinaryAccuracy(name="accuracy")]},
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_direction_accuracy",
        mode="max",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        x_train,
        {"direction": y_dir_train, "confidence": y_conf_train},
        validation_data=(x_val, {"direction": y_dir_val, "confidence": y_conf_val}),
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=[es],
        verbose=1,
    )

    pred = model.predict(x_val, batch_size=int(batch_size))
    dir_pred = (pred["direction"].reshape(-1) >= 0.5).astype(np.float32)
    dir_true = y_dir_val.reshape(-1)
    val_acc = float((dir_pred == dir_true).mean())
    avg_conf = float(np.mean(pred["confidence"].reshape(-1)))

    console.print(f"Validation directional accuracy: [bold]{val_acc*100:.2f}%[/bold]")
    console.print(f"Validation average confidence: [bold]{avg_conf*100:.2f}%[/bold]")

    model_dir = Path("trained_data") / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "buddy_tf.keras"
    meta_path = model_dir / "buddy_tf.meta.json"

    # Save model and metadata for inference/trading.
    model.save(model_path)
    meta = {
        "model_path": str(model_path),
        "csv_path": str(csv_path),
        "seq_len": int(seq_len),
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
        "confidence_scale_q95": scale,
        "val_direction_accuracy": val_acc,
        "val_avg_confidence": avg_conf,
        "trained_epochs": int(len(history.history.get("loss", []))),
        "early_stopped": bool(len(history.history.get("loss", [])) < int(epochs)),
        "live_enabled": bool(val_acc >= 0.62 and avg_conf >= 0.70),
        "live_confidence_threshold": 0.75,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    console.print(f"Saved: {model_path}")
    console.print(f"Saved: {meta_path}")


def _buddy_live_enabled_from_meta() -> tuple[bool, float]:
    """Return (live_enabled, confidence_threshold) from last Buddy training."""
    import json

    meta_path = Path("trained_data") / "models" / "buddy_tf.meta.json"
    if not meta_path.exists():
        return False, 0.75
    try:
        meta = json.loads(meta_path.read_text())
        return bool(meta.get("live_enabled", False)), float(meta.get("live_confidence_threshold", 0.75))
    except Exception:
        return False, 0.75


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

    # Add a small, conservative buffer on top of half-spread.
    # Increased to 10.0 pips to prevent nuisance cancellations in volatile markets.
    buffer_price = 10.0 * pip_size(instrument)

    if int(units) > 0:
        return float(ask + half_spread + buffer_price)
    return float(bid - half_spread - buffer_price)


def _build_integrated_engines(config: Dict[str, Any]) -> NeuralNetworkIntegrator:
    """Retired.

    Legacy integrated ML/MT/MR pipeline has been retired as part of the
    TensorFlow-only Buddy migration.
    """

    raise RuntimeError("Retired: integrated engines are no longer supported. Use `train-buddy`/`buddy`.")


def integrated_predict_once(config: Dict[str, Any]) -> Dict[str, Any]:
    raise RuntimeError("Retired: integrated prediction is no longer supported. Use `train-buddy`/`buddy`.")


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
    raise RuntimeError("Retired: integrated feature tensors are no longer supported.")


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

    price_bound = _fx_execution_guard_price_bound(policy, client, instrument=instrument, units=units)
    if price_bound is None:
        return

    result = client.create_market_order(
        instrument=instrument,
        units=units,
        stop_loss_price=stop_price,
        take_profit_price=tp_price,
        price_bound=price_bound,
    )
    state.entries_today += 1
    fxg.save_state(cfg, state)
    console.print("[bold green]Order submitted (PRACTICE).[/bold green]")
    console.print(result)


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
def train_model(config_path: str, choose_csv: bool = False) -> None:
    """Run model training with live progress dashboard.
    
    Args:
        config_path: Path to configuration file
        choose_csv: Whether to choose CSV file interactively
        
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
    config = load_config(config_path)  # removed Path() wrapper for proper string input
    # Instantiate the AI assistant using the class from ml_engine
    from ai_assistant import ai_assistant

    assistant_instance = ai_assistant(config)
    console.print("[bold blue]Launching AI Assistant...[/bold blue]")
    query = input("Enter your query: ")
    response = assistant_instance.process_query(query)
    console.print(f"[bold green]Response:[/bold green] {response}")

    # Optionally use ML Engine's AI assistant for simulation tasks
    from ml_engine_enhanced import EnhancedMLEngine

    engine_instance = EnhancedMLEngine(config)
    simulation_output = engine_instance.ai_assistant.process_query("simulate")
    console.print(
        f"[bold blue]AI Assistant Simulation Output:[/bold blue]\n{simulation_output}"
    )


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
    all_features: bool = False,
    verbose: bool = False,
) -> None:
    """Buddy: run one TF-only inference on fresh OANDA candles; optionally place a trade."""
    _configure_predict_output(verbose)
    _configure_tf_metal()

    import json
    import numpy as np
    import tensorflow as tf

    from feature_engineering import FeatureEngineering
    from fx_paper import candles_to_ohlcv_df, pip_size, position_size_units
    from oanda_practice import OandaPracticeClient

    cfg = load_config(config_path)

    # Load model + preprocessing metadata.
    meta_path = Path("trained_data") / "models" / "buddy_tf.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Missing Buddy metadata. Run: python main.py train-buddy")
    meta = json.loads(meta_path.read_text())

    model_path = checkpoint_path or meta.get("model_path")
    if not model_path:
        raise ValueError("Missing model path in metadata")

    model = tf.keras.models.load_model(model_path)
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
        enabled, _ = _buddy_live_enabled_from_meta()
        if not enabled:
            console.print(
                "[yellow]Live trading disabled[/yellow]: train Buddy and reach >=62% direction accuracy and >=70% avg confidence."
            )
            execute = False
        else:
            console.print(f"[green]Live trading enabled[/green] (confidence threshold {threshold*100:.0f}%).")

    # Fetch candles.
    client = OandaPracticeClient.from_env()
    resp = client.get_candles(instrument, granularity=granularity, count=int(candles), price="MBA")
    df = candles_to_ohlcv_df(resp)

    # Feature engineering to match training.
    if all_features:
        fe = FeatureEngineering(cfg.get("feature_engineering", {}))
        df = fe.create_features(df, include_all=True)

    numeric_df = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.ffill().bfill().fillna(0.0)

    # Build the exact feature matrix expected by the model.
    for col in feature_columns:
        if col not in numeric_df.columns:
            numeric_df[col] = 0.0
    x2d = numeric_df[feature_columns].to_numpy(dtype=np.float32, copy=True)
    if len(x2d) < seq_len_eff:
        raise ValueError(f"Need at least {seq_len_eff} rows after preprocessing; got {len(x2d)}")
    x2d = x2d[-seq_len_eff:]

    # Standardize and optional PCA.
    x_std = (x2d - mu) / np.where(sigma < 1e-6, 1.0, sigma)
    if pca_components_mat is not None and pca_mean is not None:
        x_std = (x_std - pca_mean) @ pca_components_mat.T

    x = x_std.reshape(1, seq_len_eff, -1)
    pred = model.predict(x, verbose=0)
    p_dir = float(np.asarray(pred["direction"]).reshape(-1)[0])
    p_conf = float(np.asarray(pred["confidence"]).reshape(-1)[0])

    side = "buy" if p_dir >= 0.5 else "sell"
    console.print(f"Buddy prediction: direction={p_dir:.3f} ({side}), confidence={p_conf:.3f}")

    if p_conf < threshold:
        console.print(f"[yellow]No trade[/yellow]: confidence below {threshold:.2f}")
        return

    if not execute:
        console.print("[cyan]DRY-RUN[/cyan]: would place a market order.")
        return

    # Place a conservative market order with SL/TP derived from pips.
    last_close = float(df["close"].iloc[-1])
    ps = float(pip_size(instrument))
    stop_loss_pips = float(cfg.get("buddy", {}).get("stop_loss_pips", 20.0))
    take_profit_pips = float(cfg.get("buddy", {}).get("take_profit_pips", 40.0))
    stop_distance = stop_loss_pips * ps

    units = position_size_units(
        instrument=instrument,
        equity=float(cfg.get("buddy", {}).get("equity", 10_000.0)),
        risk_per_trade_pct=float(cfg.get("buddy", {}).get("risk_per_trade_pct", 0.005)),
        stop_distance_price=stop_distance,
        price=last_close,
    )
    if side == "sell":
        units = -abs(int(units))
    else:
        units = abs(int(units))

    stop_price = last_close - stop_distance if units > 0 else last_close + stop_distance
    tp_distance = take_profit_pips * ps
    tp_price = last_close + tp_distance if units > 0 else last_close - tp_distance

    price_bound = _fx_execution_guard_price_bound(None, client, instrument=instrument, units=units)
    result = client.create_market_order(
        instrument=instrument,
        units=int(units),
        stop_loss_price=float(stop_price),
        take_profit_price=float(tp_price),
        price_bound=price_bound,
        client_tag="buddy_tf",
    )
    console.print(f"[green]Order submitted[/green]: {result}")


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
        default="./config.yaml",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed logs (default: quiet output)",
    )

    parser.add_argument(
        "--instrument",
        "-I",
        default="USD_JPY",
        help="OANDA instrument e.g. USD_JPY,EUR_USD,GBP_USD (used by buddy)",
    )
    parser.add_argument(
        "--granularity",
        "-g",
        default="M5",
        help="OANDA candle granularity, e.g. M5, M15, H1 (used by fx/fx-paper)",
    )
    parser.add_argument(
        "--candles",
        "-n",
        type=int,
        default=300,
        help="How many candles to fetch (used by fx/fx-paper)",
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
        "--execute",
        "-x",
        action="store_true",
        default=False,
        help="Enable live trading on OANDA practice account (default: disabled)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Disable live trading, only simulate orders",
    )
    parser.add_argument(
        "--all-features",
        "-A",
        action="store_true",
        help="Use all engineered features (technical indicators, statistical, time, lag, rolling) for training",
    )
    # ... add more CLI arguments as needed ...

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    command_map = {
        "train-buddy": train_buddy,
        "buddy": buddy,
        "Buddy": buddy,
    }

    try:
        if args.command == "train-buddy":
            command_map[args.command](
                args.config,
                args.csv,
                pca_components=args.pca_components,
                seq_len=args.seq_len,
                epochs=300,
                batch_size=32,
                lr=0.001,
                all_features=True,
            )
        elif args.command in {"buddy", "Buddy"}:
            should_execute = bool(args.execute) and (not args.dry_run)
            command_map[args.command](
                args.config,
                checkpoint_path=args.model_path,
                instrument=args.instrument,
                granularity=args.granularity,
                candles=args.candles,
                execute=should_execute,
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
