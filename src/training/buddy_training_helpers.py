"""
Refactored training pipeline helpers for Buddy TF model.

Extracted from _train_buddy_impl to reduce cognitive complexity.
Each function handles one logical stage of the training workflow.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from rich.panel import Panel


class _ConsoleLike(Protocol):
    def print(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        ...


def _resolve_training_csv_path(
    *,
    csv_path: str | None,
    oanda_fetch: Any,
    oanda_fetch_to_csv: Callable[[Any], str] | None,
    console: _ConsoleLike,
    market_data_dir: Path,
    default_csv_name: str,
) -> str:
    if oanda_fetch is not None:
        if oanda_fetch_to_csv is None:
            raise ValueError("oanda_fetch was provided but oanda_fetch_to_csv is None")
        return oanda_fetch_to_csv(oanda_fetch)

    if csv_path is None:
        csv_path = str(market_data_dir / default_csv_name)

    csv_p = Path(str(csv_path))
    if csv_p.exists():
        return str(csv_p)

    # Fallback to latest live dataset.
    try:
        candidates = sorted(
            market_data_dir.glob("oanda_USD_JPY_M5_live_*.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            csv_p = candidates[-1]
            console.print(f"Default CSV missing; using latest live dataset: {csv_p}")
            return str(csv_p)
    except Exception:
        pass

    return str(csv_path)


def _coerce_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _load_csv_dataframe(
    *,
    csv_path: str,
    console: _ConsoleLike,
) -> pd.DataFrame:
    t_csv = time.perf_counter()
    df = pd.read_csv(csv_path)
    elapsed = time.perf_counter() - t_csv
    console.print(Panel(
        f"[bold]Loading Training Data[/bold]\n\n"
        f"[dim]Source:[/dim] {csv_path}\n"
        f"[dim]Rows:[/dim] {int(len(df)):,}  [dim]Columns:[/dim] {int(df.shape[1])}  [dim]Time:[/dim] {elapsed:.2f}s",
        title="📄 CSV Data",
        border_style="blue",
    ))
    return df


def _normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        return df
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    return out.sort_values("time")


def _require_ohlcv_columns(df: pd.DataFrame) -> None:
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def _apply_min_volume_filter(
    *,
    df: pd.DataFrame,
    min_volume: int | None,
    console: _ConsoleLike,
) -> pd.DataFrame:
    if min_volume is None:
        return df
    try:
        before = int(len(df))
        vol = _coerce_float_series(df["volume"])
        out = df.loc[vol >= float(min_volume)].copy()
        removed = before - int(len(out))
        if removed > 0:
            console.print(
                "Filtered low-volume candles: "
                f"removed {removed} rows (min_volume={min_volume})"
            )
        return out
    except Exception as e:
        console.print(f"[yellow]Min-volume filter skipped[/yellow]: {e}")
        return df


def _apply_spread_filter(
    *,
    df: pd.DataFrame,
    spread_filter: bool,
    spread_pctl: float,
    spread_mult: float,
    console: _ConsoleLike,
) -> pd.DataFrame:
    if not spread_filter:
        return df

    if "bid_close" not in df.columns or "ask_close" not in df.columns:
        console.print(
            "[yellow]Spread filter requested but bid/ask columns missing; "
            "skipping[/yellow]"
        )
        return df

    try:
        before = int(len(df))
        bid = _coerce_float_series(df["bid_close"])
        ask = _coerce_float_series(df["ask_close"])
        spread = (ask - bid).abs()
        spread = spread.replace([np.inf, -np.inf], np.nan).dropna()
        if spread.empty:
            return df

        pctl_thr = float(np.quantile(spread.to_numpy(), float(spread_pctl)))
        med = float(np.median(spread.to_numpy()))
        thr = max(pctl_thr, float(spread_mult) * med) if med > 0 else pctl_thr

        keep_mask = (ask - bid).abs() <= thr
        out = df.loc[keep_mask].copy()

        removed = before - int(len(out))
        if removed > 0:
            console.print(
                "Filtered wide-spread candles: "
                f"removed {removed} rows "
                f"(pctl={spread_pctl}, mult={spread_mult}, thr={thr:.10g})"
            )
        return out
    except Exception as e:
        console.print(f"[yellow]Spread filter skipped[/yellow]: {e}")
        return df


def _buddy_setup_training_environment(
    *,
    timing: bool,
    force_cpu: bool,
    mixed_precision: bool,
    configure_tf_metal: Callable[..., Any],
    console: _ConsoleLike,
    setup_tracing: Callable[[], Any] | None = None,
):
    """Setup TF environment, tracing, and mixed precision."""
    configure_tf_metal(verbose=bool(timing), force_cpu=force_cpu)

    if setup_tracing is not None:
        try:
            setup_tracing()
        except Exception:
            pass

    mp_enabled = False
    if mixed_precision:
        try:
            import tensorflow as tf

            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            console.print("Mixed precision enabled: mixed_float16")
            mp_enabled = True
        except Exception as e:
            console.print(
                "[yellow]Mixed precision enable failed[/yellow]:",
                e,
            )

    return mp_enabled


def _buddy_load_and_validate_csv(
    *,
    csv_path: str | None,
    oanda_fetch: Any,
    min_volume: int | None,
    spread_filter: bool,
    spread_pctl: float,
    spread_mult: float,
    oanda_fetch_to_csv: Callable[[Any], str] | None,
    console: _ConsoleLike,
    market_data_dir: Path = Path("market_data"),
    default_csv_name: str = "oanda_USD_JPY_M5.csv",
) -> pd.DataFrame:
    """Load training CSV and apply quality filters."""
    csv_path_eff = _resolve_training_csv_path(
        csv_path=csv_path,
        oanda_fetch=oanda_fetch,
        oanda_fetch_to_csv=oanda_fetch_to_csv,
        console=console,
        market_data_dir=Path(market_data_dir),
        default_csv_name=default_csv_name,
    )

    if not Path(str(csv_path_eff)).exists():
        raise FileNotFoundError(
            "Training CSV not found: "
            f"{csv_path_eff}. "
            "Use --oanda-live to fetch, or pass --csv <path>."
        )

    df = _load_csv_dataframe(csv_path=csv_path_eff, console=console)
    df = _normalize_time_column(df)
    _require_ohlcv_columns(df)

    df = _apply_min_volume_filter(
        df=df,
        min_volume=min_volume,
        console=console,
    )
    df = _apply_spread_filter(
        df=df,
        spread_filter=spread_filter,
        spread_pctl=spread_pctl,
        spread_mult=spread_mult,
        console=console,
    )
    return df


def _fetch_paginated_candles(
    oanda: Any,
    pair: str,
    granularity: str,
    candles_per_pair: int,
    max_per_request: int = 5000,
) -> list[dict]:
    """Fetch candles with pagination for amounts > OANDA's 5000 limit."""
    pair_rows = []
    remaining = candles_per_pair
    to_time = None
    
    while remaining > 0:
        batch_size = min(remaining, max_per_request)
        params = {"granularity": granularity, "count": batch_size, "price": "MBA"}
        if to_time:
            params["to_time"] = to_time
        
        response = oanda.get_candles(pair, **params)
        candles = response.get("candles", [])
        if not candles:
            break
        
        for c in candles:
            mid = c.get("mid", {})
            pair_rows.append({
                "time": c.get("time"),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })
        
        if candles:
            to_time = candles[0].get("time")
        remaining -= len(candles)
        if len(candles) < batch_size:
            break
    
    return pair_rows


def _normalize_pair_dataframe(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Normalize OHLCV to percentage returns (instrument-agnostic)."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["time"])
    df['pair'] = pair
    
    df['open_pct'] = df['open'].pct_change()
    df['high_pct'] = (df['high'] - df['open']) / df['open']
    df['low_pct'] = (df['low'] - df['open']) / df['open']
    df['close_pct'] = df['close'].pct_change()
    
    if 'volume' in df.columns:
        vol_mean = df['volume'].rolling(100, min_periods=10).mean()
        vol_std = df['volume'].rolling(100, min_periods=10).std().clip(lower=1e-8)
        df['volume_zscore'] = (df['volume'] - vol_mean) / vol_std
    
    return df.iloc[1:].copy()


def _get_oanda_client() -> Any:
    """Initialize and return OANDA client."""
    try:
        from src.utils.oanda_practice import OandaPracticeClient
        return OandaPracticeClient.from_env()
    except ImportError as e:
        raise RuntimeError(f"Failed to connect to OANDA: {e}") from e


def _fetch_and_normalize_pair(
    oanda: Any,
    pair: str,
    granularity: str,
    candles_per_pair: int,
    console: _ConsoleLike | None,
) -> pd.DataFrame | None:
    """Fetch and normalize data for a single pair. Returns None on failure."""
    try:
        pair_rows = _fetch_paginated_candles(oanda, pair, granularity, candles_per_pair)
    except Exception as e:
        if console:
            console.print(f"  [red]✗ {pair}: {e}[/red]")
        return None
    
    if not pair_rows:
        if console:
            console.print(f"  [yellow]⚠ {pair}: no candles returned[/yellow]")
        return None
    
    df = pd.DataFrame(pair_rows)
    if len(df) < 100:
        if console:
            console.print(f"  [yellow]⚠ {pair}: insufficient data ({len(df)} rows)[/yellow]")
        return None
    
    df = _normalize_pair_dataframe(df, pair)
    if console:
        console.print(f"  [green]✓ {pair}: {len(df):,} rows[/green]")
    return df


def _combine_pair_dataframes(
    all_dfs: list[pd.DataFrame],
    console: _ConsoleLike | None,
) -> pd.DataFrame:
    """Combine, clean, and shuffle pair DataFrames."""
    if not all_dfs:
        raise ValueError("No data loaded from any pairs")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Drop columns not needed for training
    drop_cols = [c for c in ['time', 'pair'] if c in combined.columns]
    if drop_cols:
        combined = combined.drop(columns=drop_cols)
    
    # Shuffle to mix pairs (important for training)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    if console:
        console.print(f"\n  [bold]Total: {len(combined):,} rows from {len(all_dfs)} pairs[/bold]")
    
    return combined


def _load_multi_pair_data(
    pairs: list[str],
    granularity: str = "H1",
    candles_per_pair: int = 5000,
    console: _ConsoleLike | None = None,
) -> pd.DataFrame:
    """Load and concatenate data from multiple currency pairs.
    
    This enables multi-pair foundation model training by:
    1. Fetching historical data for each pair from OANDA (with pagination for >5000)
    2. Normalizing features to make them instrument-agnostic
    3. Concatenating all data with shuffle for training
    
    Args:
        pairs: List of instrument names (e.g., ["EUR_USD", "GBP_USD", "USD_JPY"])
        granularity: Candle timeframe (default: H1)
        candles_per_pair: Number of candles to fetch per pair
        console: Rich console for output
        
    Returns:
        Combined DataFrame with normalized features from all pairs
    """
    oanda = _get_oanda_client()
    
    all_dfs = []
    for pair in pairs:
        df = _fetch_and_normalize_pair(oanda, pair, granularity, candles_per_pair, console)
        if df is not None:
            all_dfs.append(df)
    
    return _combine_pair_dataframes(all_dfs, console)


# =============================================================================
# Regime-Specific LightGBM Training (Phase 3)
# =============================================================================

# Error message constants
_ERR_IMPORT_FAILED = 'Import failed'

_REGIME_FEATURE_COLS = [
    'returns_1', 'returns_2', 'returns_3', 'returns_5', 'returns_10', 'returns_20',
    'log_returns_1', 'atr_pct_5', 'atr_pct_10', 'atr_pct_14',
    'volatility_5', 'volatility_10', 'rsi_norm', 'stoch_k_norm',
    'macd_norm', 'macd_hist_norm', 'volume_ratio_5', 'volume_ratio_10', 'volume_zscore',
    'zscore_10', 'zscore_20', 'pct_rank_20', 'bb_position_20',
]


def _get_available_features(df: pd.DataFrame, preferred_cols: list[str]) -> list[str]:
    """Get available feature columns from dataframe."""
    available = [f for f in preferred_cols if f in df.columns]
    if len(available) < 10:
        # Fallback to numeric columns
        exclude = {'open', 'high', 'low', 'close', 'volume', 'direction_target'}
        available = [
            c for c in df.columns
            if df[c].dtype in ['float64', 'float32', 'int64', 'int32']
            and c not in exclude
        ][:30]
    return available


def _compute_direction_targets(
    df: pd.DataFrame,
    lookahead: int = 24,
    threshold: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute direction targets from close prices."""
    import numpy as np
    close = df['close'].values
    y = np.zeros(len(close), dtype=np.int32)
    
    for i in range(len(close) - lookahead):
        future_return = (close[i + lookahead] - close[i]) / close[i]
        if future_return > threshold:
            y[i] = 1
        elif future_return < -threshold:
            y[i] = 0
        else:
            y[i] = -1  # Ambiguous
    
    valid_mask = y >= 0
    return y, valid_mask


def _report_regime_metrics(
    console: _ConsoleLike,
    metrics: dict[str, Any],
) -> None:
    """Report regime training metrics to console."""
    console.print("\n[bold green]Regime LightGBM Training Complete[/bold green]")
    console.print("\nResults:")
    for regime, m in metrics.items():
        if regime.startswith('_'):
            continue
        status = m.get('status', 'unknown')
        if status == 'trained':
            console.print(
                f"  {regime}: val_acc={m.get('val_accuracy', 0):.3f}, "
                f"val_f1={m.get('val_f1', 0):.3f}, "
                f"n={m.get('n_train', 0)}"
            )
        elif status == 'skipped':
            console.print(f"  {regime}: [yellow]skipped[/yellow] ({m.get('reason', 'unknown')})")
        else:
            console.print(f"  {regime}: [red]{status}[/red]")


def _import_regime_modules() -> tuple[Any, ...] | None:
    """Import required modules for regime training. Returns None on failure."""
    try:
        from src.core.modular_data_loaders import (
            classify_market_regime,
            compute_normalized_features,
            get_regime_distribution,
            temporal_split,
        )
        from src.training.modular_trainers import RegimeLGBMTrainer as _RegimeLGBMTrainer
        return classify_market_regime, compute_normalized_features, get_regime_distribution, temporal_split, _RegimeLGBMTrainer
    except ImportError:
        return None


def _prepare_regime_data(
    df: pd.DataFrame,
    regimes: Any,
    available_features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare feature arrays and direction targets for regime training."""
    import numpy as np
    
    x_all = df[available_features].values.astype(np.float32)
    
    if 'direction_target' in df.columns:
        y_all = df['direction_target'].values
        regimes_arr = regimes.values[:len(x_all)]
    else:
        y_all, valid_mask = _compute_direction_targets(df)
        x_all = x_all[valid_mask]
        y_all = y_all[valid_mask]
        regimes_arr = regimes.values[valid_mask]
    
    return x_all, y_all, regimes_arr


def _log_regime_distribution(
    console: _ConsoleLike | None,
    distribution: dict[str, int],
    regimes_len: int,
) -> None:
    """Log regime distribution to console."""
    if console is None:
        return
    console.print("\n  Regime Distribution:")
    for regime_name, count in sorted(distribution.items()):
        pct = count / regimes_len * 100
        console.print(f"    {regime_name}: {count:,} ({pct:.1f}%)")


def train_regime_lgbm_models(
    df: pd.DataFrame,
    instrument: str,
    console: _ConsoleLike | None = None,
    model_dir: str = "trained_data/models",
    min_samples_per_regime: int = 100,
) -> dict[str, Any]:
    """
    Train 5 separate LightGBM models, one per market regime.
    
    This function:
    1. Computes regime labels for training data using classify_market_regime()
    2. Splits data by regime
    3. Trains a specialized LightGBM model for each regime
    4. Saves models to trained_data/models/{instrument}/lgbm_regime_{REGIME}.pkl
    
    Args:
        df: DataFrame with OHLCV data and features
        instrument: Currency pair name (e.g., 'EUR_USD')
        console: Rich console for output
        model_dir: Base directory for model artifacts
        min_samples_per_regime: Minimum samples required to train a regime model
    
    Returns:
        Dict with training results and saved model paths
    """
    modules = _import_regime_modules()
    if modules is None:
        if console:
            console.print("[red]Failed to import required modules[/red]")
        return {'status': 'error', 'error': _ERR_IMPORT_FAILED}
    
    classify_market_regime, compute_normalized_features, get_regime_distribution, temporal_split, regime_lgbm_trainer_cls = modules
    
    if console:
        console.print("\n[bold blue]Training Regime-Specific LightGBM Models[/bold blue]")
        console.print(f"Instrument: {instrument}")
    
    # Compute normalized features if not present
    if 'returns_1' not in df.columns:
        if console:
            console.print("  Computing normalized features...")
        df = compute_normalized_features(df)
    
    # Classify market regimes
    if console:
        console.print("  Classifying market regimes...")
    regimes, _ = classify_market_regime(df, return_confidence=True)
    
    # Log regime distribution
    distribution = get_regime_distribution(regimes)
    _log_regime_distribution(console, distribution, len(regimes))
    
    # Prepare features
    available_features = _get_available_features(df, _REGIME_FEATURE_COLS)
    if console:
        console.print(f"\n  Using {len(available_features)} features")
    
    x_all, y_all, regimes_arr = _prepare_regime_data(df, regimes, available_features)
    
    # Temporal split (70/20/10)
    train_idx, val_idx, test_idx = temporal_split(len(x_all), 0.7, 0.2, 0.1)
    
    x_train, x_val = x_all[train_idx], x_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    regimes_train, regimes_val = regimes_arr[train_idx], regimes_arr[val_idx]
    
    if console:
        console.print(f"\n  Train: {len(x_train):,}, Val: {len(x_val):,}, Test: {len(test_idx):,}")
        console.print("\n  Training regime models...")
    
    # Train regime-specific models
    trainer = regime_lgbm_trainer_cls()
    metrics = trainer.train_regime_models(
        X_train=x_train,
        y_train=y_train,
        regimes_train=regimes_train,
        X_val=x_val,
        y_val=y_val,
        regimes_val=regimes_val,
        feature_names=available_features,
        min_samples_per_regime=min_samples_per_regime,
    )
    
    # Save models
    if console:
        console.print(f"\n  Saving models to {model_dir}/{instrument}/...")
    saved_paths = trainer.save(model_dir, instrument)
    
    # Report results
    if console:
        _report_regime_metrics(console, metrics)
    
    return {
        'status': 'success',
        'instrument': instrument,
        'metrics': metrics,
        'saved_paths': {k: str(v) for k, v in saved_paths.items()},
        'regime_distribution': distribution,
        'n_train': len(x_train),
        'n_val': len(x_val),
        'n_features': len(available_features),
    }


_STRESS_TEST_FEATURE_COLS = [
    'returns_1', 'returns_2', 'returns_3', 'returns_5', 'returns_10', 'returns_20',
    'log_returns_1', 'atr_pct_5', 'atr_pct_10', 'atr_pct_14',
    'volatility_5', 'volatility_10', 'rsi_norm', 'stoch_k_norm',
    'macd_norm', 'macd_hist_norm', 'volume_ratio_5', 'volume_ratio_10', 'volume_zscore',
]

_REGIME_NAME_MAP = {
    0: 'STRONG_TREND',
    1: 'WEAK_TREND',
    2: 'CHOP',
    3: 'MEAN_REVERT',
    4: 'BREAKOUT',
}


def _import_stress_test_modules() -> tuple[Any, ...] | None:
    """Import required modules for stress testing. Returns None on failure."""
    try:
        from src.core.modular_data_loaders import (
            classify_market_regime,
            compute_normalized_features,
        )
        from src.training.modular_trainers import RegimeLGBMTrainer as _RegimeLGBMTrainer
        from src.training.walkforward_validation import (
            stress_test_regime,
            stress_test_regime_transitions,
        )
        return (
            classify_market_regime,
            compute_normalized_features,
            _RegimeLGBMTrainer,
            stress_test_regime,
            stress_test_regime_transitions,
        )
    except ImportError:
        return None


def _get_regime_predictions(
    trainer: Any,
    X_test: np.ndarray,
    regimes_test: np.ndarray,
) -> np.ndarray:
    """Get predictions from regime-specific models."""
    import numpy as np
    
    y_pred = []
    for i, regime in enumerate(regimes_test):
        regime_name = _REGIME_NAME_MAP.get(regime, 'WEAK_TREND')
        try:
            result = trainer.predict(X_test[i:i+1], regime=regime_name)
            y_pred.append(result['direction'])
        except Exception:
            y_pred.append(0)
    
    return np.array(y_pred)


def _compute_true_directions(close_prices: np.ndarray) -> np.ndarray:
    """Compute actual direction from close prices."""
    import numpy as np
    
    y_true = (np.diff(close_prices) > 0).astype(int)
    return np.concatenate([y_true, [0]])  # Pad last


def _report_stress_test_results(
    console: _ConsoleLike,
    regime_results: dict[str, Any],
    transition_results: dict[str, Any],
) -> None:
    """Report stress test results to console."""
    console.print("\n[bold]Per-Regime Performance:[/bold]")
    for regime, metrics in regime_results.items():
        if regime.startswith('_'):
            continue
        acc = metrics.get('accuracy', 0)
        target = metrics.get('target', 0.5)
        met = "✓" if metrics.get('target_met', False) else "✗"
        console.print(f"  {regime}: {acc:.1%} (target: {target:.0%}) {met}")
    
    console.print("\n[bold]Transition Performance:[/bold]")
    if 'before_transition' in transition_results:
        console.print(f"  Before: {transition_results['before_transition'].get('accuracy', 0):.1%}")
    if 'after_transition' in transition_results:
        console.print(f"  After: {transition_results['after_transition'].get('accuracy', 0):.1%}")
    if 'stable' in transition_results:
        console.print(f"  Stable: {transition_results['stable'].get('accuracy', 0):.1%}")


def run_regime_stress_test(
    df: pd.DataFrame,
    model_dir: str,
    instrument: str,
    console: _ConsoleLike | None = None,
) -> dict[str, Any]:
    """
    Run stress tests on regime-specific models.
    
    Tests model performance per regime and around regime transitions.
    
    Args:
        df: DataFrame with OHLCV data
        model_dir: Directory containing trained models
        instrument: Currency pair name
        console: Rich console for output
    
    Returns:
        Dict with stress test results
    """
    import numpy as np
    
    modules = _import_stress_test_modules()
    if modules is None:
        if console:
            console.print("[red]Failed to import modules[/red]")
        return {'status': 'error', 'error': _ERR_IMPORT_FAILED}
    
    (classify_market_regime, compute_normalized_features,
     regime_lgbm_trainer_cls, stress_test_regime, stress_test_regime_transitions) = modules
    
    if console:
        console.print("\n[bold blue]Running Regime Stress Tests[/bold blue]")
    
    # Compute features and regimes
    if 'returns_1' not in df.columns:
        df = compute_normalized_features(df)
    
    regimes, _ = classify_market_regime(df, return_confidence=True)
    
    # Load regime models
    trainer = regime_lgbm_trainer_cls()
    loaded = trainer.load(model_dir, instrument)
    
    if not loaded:
        if console:
            console.print("[yellow]No regime models found to test[/yellow]")
        return {'status': 'no_models'}
    
    if console:
        console.print(f"  Loaded models: {loaded}")
    
    # Prepare test data (use last 10%)
    test_start = int(len(df) * 0.9)
    available_features = [f for f in _STRESS_TEST_FEATURE_COLS if f in df.columns]
    
    X_test = df[available_features].iloc[test_start:].values.astype(np.float32)
    regimes_test = regimes.iloc[test_start:].values
    
    # Get predictions and true labels
    y_pred = _get_regime_predictions(trainer, X_test, regimes_test)
    y_true = _compute_true_directions(df['close'].values[test_start:])
    
    # Truncate to same length
    min_len = min(len(y_pred), len(y_true), len(regimes_test))
    y_pred, y_true, regimes_test = y_pred[:min_len], y_true[:min_len], regimes_test[:min_len]
    
    # Run stress tests
    regime_results = stress_test_regime(y_pred, y_true, regimes_test)
    transition_results = stress_test_regime_transitions(y_pred, y_true, regimes_test)
    
    if console:
        _report_stress_test_results(console, regime_results, transition_results)
    
    return {
        'status': 'success',
        'regime_results': regime_results,
        'transition_results': transition_results,
    }


# =============================================================================
# Joint Multi-Pair Training Helper Functions
# =============================================================================

def _fetch_instrument_data(
    instrument: str,
    granularity: str,
    candles: int,
    oanda_client: Any,
    console: _ConsoleLike | None = None,
) -> pd.DataFrame | None:
    """Fetch and normalize data for a single instrument."""
    from src.core.modular_data_loaders import compute_normalized_features
    
    try:
        all_rows = _fetch_paginated_candles(oanda_client, instrument, granularity, candles)
        
        if not all_rows:
            if console:
                console.print(f"    [yellow]No data for {instrument}[/yellow]")
            return None
        
        df = pd.DataFrame(all_rows)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        df.set_index('time', inplace=True)
        df = compute_normalized_features(df)
        
        if console:
            console.print(f"    [green]✓ {len(df)} rows loaded[/green]")
        return df
        
    except Exception as e:
        if console:
            console.print(f"    [red]Error: {e}[/red]")
        return None


def _combine_dfs_for_rl(
    dfs: dict[str, pd.DataFrame],
    console: _ConsoleLike | None,
) -> pd.DataFrame:
    """Combine DataFrames for RL training."""
    if console:
        console.print(f"  [dim]Combining {len(dfs)} DataFrames...[/dim]")
    combined_df = pd.concat([df.reset_index() for df in dfs.values()], ignore_index=True)
    if console:
        console.print(f"  [dim]Combined shape: {combined_df.shape}[/dim]")
    return combined_df


def _get_rl_feature_cols(
    combined_df: pd.DataFrame,
    console: _ConsoleLike | None,
) -> list[str]:
    """Get feature columns for RL training, with fallback."""
    exclude_cols = {'time', 'open', 'high', 'low', 'close', 'volume', 'pair'}
    feature_cols = [c for c in combined_df.columns if c not in exclude_cols]
    if console:
        console.print(f"  [dim]Feature columns: {len(feature_cols)}[/dim]")
    
    if len(feature_cols) < 5:
        fallback_cols = ['returns_1', 'returns_2', 'returns_5', 'atr_pct_14', 'rsi_norm']
        feature_cols = [c for c in fallback_cols if c in combined_df.columns]
        if console:
            console.print(f"  [dim]Using fallback features: {len(feature_cols)}[/dim]")
    
    return feature_cols


def _prepare_rl_arrays(
    combined_df: pd.DataFrame,
    feature_cols: list[str],
    console: _ConsoleLike | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare numpy arrays for RL training."""
    if console:
        console.print("  [dim]Preparing RL data arrays...[/dim]")
    
    rl_features = combined_df[feature_cols].fillna(0).values.astype(np.float32)
    rl_prices = combined_df['close'].values.astype(np.float32)
    n_samples = len(rl_features)
    rl_predictions = np.column_stack([
        np.full(n_samples, 0.5),
        np.full(n_samples, 0.5),
    ]).astype(np.float32)
    
    if console:
        console.print(f"  [dim]Features shape: {rl_features.shape}, Prices: {len(rl_prices)} samples[/dim]")
    
    return rl_features, rl_predictions, rl_prices


def _train_rl_position_sizer(
    dfs: dict[str, pd.DataFrame],
    model_dir: str,
    console: _ConsoleLike | None = None,
) -> dict[str, Any]:
    """Train RL position sizer on combined data."""
    from pathlib import Path
    
    try:
        from rl_position_sizing import train_rl_position_sizer
    except ImportError as e:
        return {'status': 'skipped', 'reason': f'RL dependencies not available: {e}'}
    
    if console:
        console.print("\n[bold]Training RL Position Sizer...[/bold]")
    
    try:
        combined_df = _combine_dfs_for_rl(dfs, console)
        feature_cols = _get_rl_feature_cols(combined_df, console)
        
        if len(feature_cols) < 5 or 'close' not in combined_df.columns:
            if console:
                console.print("  [yellow]⚠ RL sizer skipped (insufficient features)[/yellow]")
            return {'status': 'skipped', 'reason': 'Insufficient features for RL training'}
        
        rl_features, rl_predictions, rl_prices = _prepare_rl_arrays(combined_df, feature_cols, console)
        
        if console:
            console.print("  [dim]Starting PPO training (this may take a few minutes)...[/dim]")
        
        rl_sizer = train_rl_position_sizer(
            features=rl_features,
            ensemble_predictions=rl_predictions,
            prices=rl_prices,
            verbose=1,
        )
        
        if rl_sizer is None:
            if console:
                console.print("  [yellow]⚠ RL sizer training returned None[/yellow]")
            return {'status': 'skipped', 'reason': 'No model returned'}
        
        rl_save_path = Path(model_dir) / "rl_position_sizer.pkl"
        rl_sizer.save(str(rl_save_path))
        if console:
            console.print("  [green]✓ RL Position Sizer trained[/green]")
        return {'status': 'success', 'path': str(rl_save_path)}
        
    except Exception as e:
        if console:
            console.print(f"  [yellow]⚠ RL sizer training failed: {e}[/yellow]")
        return {'status': 'failed', 'error': str(e)}


def _collect_volatility_data(
    dfs: dict[str, pd.DataFrame],
    load_volatility_regime_data: Callable,
    console: _ConsoleLike | None,
) -> tuple[list, list, list, list, list | None]:
    """Collect volatility regime data from all instruments."""
    vol_x_train, vol_y_train = [], []
    vol_x_val, vol_y_val = [], []
    feature_names = None
    
    for instrument, df in dfs.items():
        try:
            vol_data = load_volatility_regime_data(df)
            if vol_data and vol_data.get('X_train') is not None:
                vol_x_train.append(vol_data['X_train'])
                vol_y_train.append(vol_data['y_train'])
                vol_x_val.append(vol_data['X_val'])
                vol_y_val.append(vol_data['y_val'])
                if feature_names is None:
                    feature_names = vol_data.get('feature_names')
        except Exception as e:
            if console:
                console.print(f"  [dim]Skipping {instrument} for TCN: {e}[/dim]")
    
    return vol_x_train, vol_y_train, vol_x_val, vol_y_val, feature_names


def _stack_volatility_arrays(
    vol_x_train: list,
    vol_y_train: list,
    vol_x_val: list,
    vol_y_val: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack volatility data lists into numpy arrays."""
    import numpy as np
    return (
        np.vstack(vol_x_train),
        np.concatenate(vol_y_train),
        np.vstack(vol_x_val),
        np.concatenate(vol_y_val),
    )


def _train_tcn_volatility(
    dfs: dict[str, pd.DataFrame],
    model_dir: str,
    config: Any,
    console: _ConsoleLike | None = None,
) -> dict[str, Any]:
    """Train TCN volatility regime model on combined data."""
    from pathlib import Path
    
    try:
        from src.training.modular_trainers import TCNTrainer
        from src.core.modular_data_loaders import load_volatility_regime_data
    except ImportError as e:
        return {'status': 'skipped', 'reason': f'TCN dependencies not available: {e}'}
    
    if console:
        console.print("\n[bold]Training TCN Volatility Regime Model...[/bold]")
    
    try:
        vol_x_train, vol_y_train, vol_x_val, vol_y_val, feature_names = _collect_volatility_data(
            dfs, load_volatility_regime_data, console
        )
        
        if not vol_x_train:
            if console:
                console.print("  [yellow]⚠ TCN skipped (no data)[/yellow]")
            return {'status': 'skipped', 'reason': 'No volatility data available'}
        
        tcn_x_train, tcn_y_train, tcn_x_val, tcn_y_val = _stack_volatility_arrays(
            vol_x_train, vol_y_train, vol_x_val, vol_y_val
        )
        
        tcn_trainer = TCNTrainer(config)
        tcn_save_path = Path(model_dir) / "joint" / "tcn_volatility_regime.keras"
        warm_start_path = str(tcn_save_path) if tcn_save_path.exists() else None
        
        tcn_metrics = tcn_trainer.train(
            tcn_x_train, tcn_y_train, tcn_x_val, tcn_y_val,
            feature_names=feature_names,
            warm_start_path=warm_start_path,
            instrument="joint",
        )
        
        tcn_trainer.save(str(tcn_save_path))
        
        if console:
            acc = tcn_metrics.get('val_accuracy', tcn_metrics.get('accuracy', 0))
            console.print(f"  [green]✓ TCN Volatility trained (val_acc={acc:.2%})[/green]")
        return {'status': 'success', 'metrics': tcn_metrics, 'path': str(tcn_save_path)}
        
    except Exception as e:
        if console:
            console.print(f"  [yellow]⚠ TCN training failed: {e}[/yellow]")
        return {'status': 'failed', 'error': str(e)}


def _train_meta_labeler(
    trainer: Any,
    model_dir: str,
    console: _ConsoleLike | None = None,
) -> dict[str, Any]:
    """Train meta-labeler using direction trainer predictions."""
    from pathlib import Path
    
    try:
        from src.training.meta_labeling import MetaLabeler, MetaLabelingConfig
    except ImportError as e:
        return {'status': 'skipped', 'reason': f'Meta-labeling dependencies not available: {e}'}
    
    if console:
        console.print("\n[bold]Training Meta-Labeler...[/bold]")
    
    try:
        training_data = getattr(trainer, 'training_data', None)
        direction_trainer = getattr(trainer, 'direction_trainer', None)
        
        has_data = (
            training_data is not None
            and 'direction' in training_data
            and training_data['direction'].get('X_train') is not None
        )
        
        if direction_trainer is None or not has_data:
            if console:
                console.print("  [yellow]⚠ Meta-labeler skipped (no direction trainer/data)[/yellow]")
            return {'status': 'skipped', 'reason': 'Direction trainer or training data not available'}
        
        import numpy as np
        
        x_train = training_data['direction']['X_train']
        y_train = training_data['direction']['y_train']
        x_val = training_data['direction']['X_val']
        y_val = training_data['direction']['y_val']
        
        train_preds = direction_trainer.predict(x_train)
        val_preds = direction_trainer.predict(x_val)
        
        train_pred_binary = (train_preds > 0.5).astype(int).flatten()
        val_pred_binary = (val_preds > 0.5).astype(int).flatten()
        train_conf = np.abs(train_preds - 0.5).flatten() * 2
        val_conf = np.abs(val_preds - 0.5).flatten() * 2
        
        meta_y_train = (train_pred_binary == y_train.flatten()).astype(int)
        meta_y_val = (val_pred_binary == y_val.flatten()).astype(int)
        
        meta_x_train = np.column_stack([train_conf, train_preds.flatten()])
        meta_x_val = np.column_stack([val_conf, val_preds.flatten()])
        
        labeler = MetaLabeler(MetaLabelingConfig())
        meta_metrics = labeler.train(meta_x_train, meta_y_train, meta_x_val, meta_y_val)
        
        meta_save_path = Path(model_dir) / "joint" / "meta_labeler.pkl"
        labeler.save(meta_save_path)
        
        val_acc = meta_metrics.get('val_accuracy', meta_metrics.get('meta_val_accuracy', 0))
        if console:
            console.print(f"  [green]✓ Meta-Labeler trained (val_acc={val_acc:.2%})[/green]")
        return {'status': 'success', 'metrics': meta_metrics, 'path': str(meta_save_path)}
        
    except Exception as e:
        if console:
            console.print(f"  [yellow]⚠ Meta-labeler failed: {e}[/yellow]")
        return {'status': 'failed', 'error': f'Prediction failed: {e}'}


def _format_metric_value(value: Any) -> str:
    """Format a metric value for display."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_joint_model_metrics(console: _ConsoleLike, joint_metrics: dict[str, Any]) -> None:
    """Print joint model metrics section."""
    console.print("\n[bold]Joint Model Metrics:[/bold]")
    for model_type, metrics in joint_metrics.items():
        console.print(f"  {model_type}:")
        for k, v in metrics.items():
            console.print(f"    {k}: {_format_metric_value(v)}")


def _print_per_instrument_results(
    console: _ConsoleLike, per_instrument_metrics: dict[str, Any]
) -> None:
    """Print per-instrument results section."""
    console.print("\n[bold]Per-Instrument Results:[/bold]")
    for instrument, metrics in per_instrument_metrics.items():
        acc = metrics.get('direction_accuracy', 0)
        console.print(f"  {instrument}: accuracy={acc:.2%}")


def _print_auxiliary_model_results(
    console: _ConsoleLike,
    rl_result: dict[str, Any] | None,
    tcn_result: dict[str, Any] | None,
    meta_result: dict[str, Any] | None,
) -> None:
    """Print auxiliary model training results."""
    if rl_result:
        console.print(f"\n[bold]RL Position Sizer:[/bold] {rl_result.get('status', 'unknown')}")
    if tcn_result:
        console.print(f"[bold]TCN Volatility Regime:[/bold] {tcn_result.get('status', 'unknown')}")
    if meta_result:
        console.print(f"[bold]Meta-Labeler:[/bold] {meta_result.get('status', 'unknown')}")


def _print_saved_model_paths(
    console: _ConsoleLike,
    fine_tune_results: dict[str, Any],
    joint_save_dir: str,
    model_dir: str,
) -> None:
    """Print saved model paths section."""
    console.print("\n[bold]Models saved to:[/bold]")
    console.print(f"  Joint: {joint_save_dir}")
    for instrument in fine_tune_results:
        console.print(f"  {instrument}: {model_dir}/{instrument}")


def _report_joint_training_results(
    console: _ConsoleLike,
    joint_metrics: dict[str, Any],
    per_instrument_metrics: dict[str, Any],
    fine_tune_results: dict[str, Any],
    rl_result: dict[str, Any] | None,
    tcn_result: dict[str, Any] | None,
    meta_result: dict[str, Any] | None,
    joint_save_dir: str,
    model_dir: str,
) -> None:
    """Report joint training results to console."""
    console.print("\n[bold green]═" * 60 + "[/bold green]")
    console.print("[bold green]JOINT TRAINING COMPLETE[/bold green]")
    console.print("[bold green]═" * 60 + "[/bold green]")
    
    _print_joint_model_metrics(console, joint_metrics)
    _print_per_instrument_results(console, per_instrument_metrics)
    
    if fine_tune_results:
        console.print("\n[bold]Fine-Tuning Results:[/bold]")
        for instrument, result in fine_tune_results.items():
            console.print(f"  {instrument}: {result.get('status', 'unknown')}")
    
    _print_auxiliary_model_results(console, rl_result, tcn_result, meta_result)
    
    if fine_tune_results:
        _print_saved_model_paths(console, fine_tune_results, joint_save_dir, model_dir)
    else:
        console.print("\n[bold]Models saved to:[/bold]")
        console.print(f"  Joint: {joint_save_dir}")


# =============================================================================
# Joint Multi-Pair Training (Phase 4)
# =============================================================================

def _import_joint_training_modules() -> tuple[Any, Any, Any] | None:
    """Import required modules for joint training. Returns None on failure."""
    try:
        from src.training.modular_trainers import JointMultiPairTrainer, TrainerConfig
        from src.utils.oanda_practice import OandaPracticeClient
        return JointMultiPairTrainer, TrainerConfig, OandaPracticeClient
    except ImportError:
        return None


def _print_joint_training_header(
    console: _ConsoleLike,
    instruments: list[str],
) -> None:
    """Print joint training header to console."""
    console.print("\n[bold blue]═" * 60 + "[/bold blue]")
    console.print("[bold blue]JOINT MULTI-PAIR ENSEMBLE TRAINING (Phase 4)[/bold blue]")
    console.print(f"[bold blue]Instruments: {', '.join(instruments)}[/bold blue]")
    console.print("[bold blue]═" * 60 + "[/bold blue]")


def _fetch_all_instrument_data(
    instruments: list[str],
    granularity: str,
    candles: int,
    oanda_client: Any,
    console: _ConsoleLike | None,
) -> dict[str, pd.DataFrame]:
    """Fetch data for all instruments."""
    dfs: dict[str, pd.DataFrame] = {}
    for instrument in instruments:
        if console:
            console.print(f"\n  Fetching {instrument} ({candles} candles)...")
        df = _fetch_instrument_data(instrument, granularity, candles, oanda_client, console)
        if df is not None:
            dfs[instrument] = df
    return dfs


def _train_joint_models(
    dfs: dict[str, pd.DataFrame],
    trainer: Any,
    joint_save_dir: str,
    console: _ConsoleLike | None,
) -> dict[str, Any] | None:
    """Train joint models. Returns None on failure."""
    if console:
        console.print("\n[bold]Training Joint Models...[/bold]")
    
    try:
        return trainer.train(dfs=dfs, instruments=list(dfs.keys()), save_dir=joint_save_dir)
    except Exception as e:
        if console:
            console.print(f"[red]Training failed: {e}[/red]")
        return None


def _evaluate_per_instrument(
    trainer: Any,
    dfs: dict[str, pd.DataFrame],
    console: _ConsoleLike | None,
) -> dict[str, Any]:
    """Evaluate per-instrument performance."""
    if console:
        console.print("\n[bold]Evaluating Per-Instrument Performance...[/bold]")
    
    try:
        return trainer.evaluate_per_instrument(dfs=dfs, instruments=list(dfs.keys()))
    except Exception as e:
        if console:
            console.print(f"[yellow]Evaluation failed: {e}[/yellow]")
        return {}


def _build_joint_training_result(
    dfs: dict[str, pd.DataFrame],
    joint_metrics: dict[str, Any],
    per_instrument_metrics: dict[str, Any],
    fine_tune_results: dict[str, Any],
    tcn_result: dict[str, Any],
    meta_result: dict[str, Any],
    joint_save_dir: str,
) -> dict[str, Any]:
    """Build the result dictionary for joint training."""
    return {
        'status': 'success',
        'instruments': list(dfs.keys()),
        'joint_metrics': joint_metrics,
        'per_instrument_metrics': per_instrument_metrics,
        'fine_tune_results': fine_tune_results,
        'rl_result': None,  # RL training removed - use train-rl-sizer command
        'tcn_result': tcn_result,
        'meta_result': meta_result,
        'joint_save_dir': joint_save_dir,
        'n_instruments': len(dfs),
    }


def train_joint_multi_pair_ensemble(
    instruments: list[str],
    granularity: str = "H1",
    candles: int = 15000,
    model_dir: str = "trained_data/models",
    fine_tune: bool = True,
    fine_tune_threshold: float = 0.05,
    console: _ConsoleLike | None = None,
    oanda_client: Any | None = None,
) -> dict[str, Any]:
    """
    Train modular ensemble with joint multi-pair data.
    
    Models are saved to:
    - Joint models: trained_data/models/joint/
    - Fine-tuned models: trained_data/models/{instrument}/
    """
    from pathlib import Path
    
    modules = _import_joint_training_modules()
    if modules is None:
        if console:
            console.print("[red]Failed to import required modules[/red]")
        return {'status': 'error', 'error': _ERR_IMPORT_FAILED}
    
    joint_trainer_cls, trainer_config_cls, oanda_client_cls = modules
    
    if console:
        _print_joint_training_header(console, instruments)
    
    # 1. Fetch data for all instruments
    if oanda_client is None:
        oanda_client = oanda_client_cls.from_env()
    
    dfs = _fetch_all_instrument_data(instruments, granularity, candles, oanda_client, console)
    
    if not dfs:
        if console:
            console.print("[red]No data loaded for any instrument[/red]")
        return {'status': 'error', 'error': 'No data loaded'}
    
    # 2. Train joint models
    config = trainer_config_cls()
    trainer = joint_trainer_cls(config)
    joint_save_dir = str(Path(model_dir) / "joint")
    
    joint_metrics = _train_joint_models(dfs, trainer, joint_save_dir, console)
    if joint_metrics is None:
        return {'status': 'error', 'error': 'Training failed'}
    
    # 3. Evaluate per-instrument performance
    per_instrument_metrics = _evaluate_per_instrument(trainer, dfs, console)
    
    # 4. Fine-tune underperforming pairs
    fine_tune_results = _run_fine_tuning(
        trainer, dfs, per_instrument_metrics, fine_tune, fine_tune_threshold, model_dir, console
    )
    
    # 5. Train auxiliary models (RL sizer removed - train separately via train-rl-sizer)
    tcn_result = _train_tcn_volatility(dfs, model_dir, config, console)
    meta_result = _train_meta_labeler(trainer, model_dir, console)
    
    # 6. Report results
    if console:
        _report_joint_training_results(
            console, joint_metrics, per_instrument_metrics, fine_tune_results,
            None, tcn_result, meta_result, joint_save_dir, model_dir
        )
    
    return _build_joint_training_result(
        dfs, joint_metrics, per_instrument_metrics, fine_tune_results,
        tcn_result, meta_result, joint_save_dir
    )


def _fine_tune_single_instrument(
    trainer: Any,
    df: pd.DataFrame,
    instrument: str,
    model_dir: str,
    console: _ConsoleLike | None,
) -> dict[str, Any]:
    """Fine-tune a single instrument. Returns result dict."""
    if console:
        console.print(f"\n  Fine-tuning {instrument}...")
    
    try:
        ft_metrics = trainer.fine_tune_for_pair(
            df=df, instrument=instrument, save_dir=model_dir
        )
        if console:
            console.print("    [green]✓ Fine-tuned[/green]")
        return {'status': 'success', 'metrics': ft_metrics}
    except Exception as e:
        if console:
            console.print(f"    [red]Failed: {e}[/red]")
        return {'status': 'failed', 'error': str(e)}


def _run_fine_tuning(
    trainer: Any,
    dfs: dict[str, pd.DataFrame],
    per_instrument_metrics: dict[str, Any],
    fine_tune: bool,
    fine_tune_threshold: float,
    model_dir: str,
    console: _ConsoleLike | None,
) -> dict[str, Any]:
    """Run fine-tuning for underperforming instruments."""
    if not fine_tune or not per_instrument_metrics:
        return {}
    
    if console:
        console.print("\n[bold]Checking Fine-Tuning Needs...[/bold]")
    
    decisions = trainer.should_fine_tune(performance_threshold=fine_tune_threshold)
    fine_tune_results = {}
    
    for instrument, needs_tuning in decisions.items():
        if needs_tuning and instrument in dfs:
            fine_tune_results[instrument] = _fine_tune_single_instrument(
                trainer, dfs[instrument], instrument, model_dir, console
            )
    
    return fine_tune_results


def train_with_walkforward_validation(
    trainer_class: type,
    trainer_config: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str] | None = None,
    w_train: np.ndarray | None = None,
    w_val: np.ndarray | None = None,
    warm_start_path: str | None = None,
    instrument: str = "UNKNOWN",
    wf_config: dict[str, Any] | None = None,
    console: _ConsoleLike | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Train a model with walk-forward cross-validation.
    
    This function wraps the standard training process with walk-forward validation,
    providing more robust out-of-sample performance estimates.
    
    Args:
        trainer_class: Trainer class to use (e.g., TransformerDirectionTrainer)
        trainer_config: Configuration object for the trainer
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (combined with train for WF splits)
        y_val: Validation labels
        feature_names: Optional feature names
        w_train: Optional sample weights for training
        w_val: Optional sample weights for validation
        warm_start_path: Optional path to warm-start model
        instrument: Instrument name for logging
        wf_config: Walk-forward configuration dict from YAML
        console: Rich console for output
    
    Returns:
        Tuple of (best_trainer, aggregated_metrics)
    """
    from src.training.walkforward_validation import (
        WalkForwardConfig,
        WalkForwardValidator,
    )
    
    # Load walk-forward configuration
    if wf_config is None or not wf_config.get('enabled', True):
        # Walk-forward disabled, use standard training
        trainer = trainer_class(trainer_config)
        metrics = trainer.train(
            X_train, y_train, X_val, y_val,
            feature_names=feature_names,
            w_train=w_train,
            w_val=w_val,
            warm_start_path=warm_start_path,
            instrument=instrument,
        )
        return trainer, metrics
    
    # Parse walk-forward config
    wf_cfg = WalkForwardConfig.from_dict(wf_config)
    
    if console:
        console.print(Panel(
            f"[bold]Walk-Forward Cross-Validation[/bold]\n\n"
            f"[dim]Mode:[/dim] {wf_cfg.mode} (sliding window)\n"
            f"[dim]Splits:[/dim] {wf_cfg.n_splits} folds\n"
            f"[dim]Train Size:[/dim] {wf_cfg.train_size*100:.0f}%\n"
            f"[dim]Gap:[/dim] {wf_cfg.gap} samples\n"
            f"[dim]Retrain per fold:[/dim] {wf_cfg.retrain_per_fold}",
            title="📊 Robust Validation",
            border_style="cyan",
        ))
    
    # Combine train and val for walk-forward splitting
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    if w_train is not None and w_val is not None:
        w_combined = np.concatenate([w_train, w_val], axis=0)
    else:
        w_combined = None
    
    # Create validator
    validator = WalkForwardValidator(
        n_splits=wf_cfg.n_splits,
        train_size=wf_cfg.train_size,
        val_size=wf_cfg.val_size,
        test_size=wf_cfg.test_size,
        gap=wf_cfg.gap,
        mode=wf_cfg.mode,
        min_train_size=wf_cfg.min_train_size,
    )
    
    # Train on each fold
    fold_trainers = []
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(validator.split(X_combined)):
        if console:
            console.print(f"\n[cyan]Fold {fold_idx + 1}/{wf_cfg.n_splits}[/cyan]: "
                         f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        # Split data for this fold
        X_train_fold = X_combined[train_idx]
        y_train_fold = y_combined[train_idx]
        X_val_fold = X_combined[val_idx]
        y_val_fold = y_combined[val_idx]
        
        w_train_fold = w_combined[train_idx] if w_combined is not None else None
        w_val_fold = w_combined[val_idx] if w_combined is not None else None
        
        if wf_cfg.retrain_per_fold:
            # Create fresh trainer for this fold
            fold_trainer = trainer_class(trainer_config)
            
            # Train on this fold
            fold_metrics_dict = fold_trainer.train(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                feature_names=feature_names,
                w_train=w_train_fold,
                w_val=w_val_fold,
                warm_start_path=warm_start_path,
                instrument=instrument,
                data_range=f"fold_{fold_idx+1}",
            )
            
            fold_trainers.append(fold_trainer)
            fold_metrics.append(fold_metrics_dict)
            
            if console:
                val_acc = fold_metrics_dict.get('val_accuracy', 0.0)
                console.print(f"  [green]✓ Fold {fold_idx + 1} complete: val_accuracy={val_acc:.1%}[/green]")
        else:
            # Use same trainer for all folds (not recommended)
            if fold_idx == 0:
                fold_trainer = trainer_class(trainer_config)
                fold_trainers.append(fold_trainer)
            else:
                fold_trainer = fold_trainers[0]
            
            # Evaluate on this fold without retraining
            # Just collect metrics
            fold_metrics_dict = {'val_accuracy': 0.0}  # Placeholder
            fold_metrics.append(fold_metrics_dict)
    
    # Aggregate results based on strategy
    if wf_cfg.aggregate_method == "best":
        # Select best fold based on validation accuracy
        best_idx = max(range(len(fold_metrics)), 
                      key=lambda i: fold_metrics[i].get('val_accuracy', 0.0))
        best_trainer = fold_trainers[best_idx]
        best_metrics = fold_metrics[best_idx]
        
        if console:
            console.print(f"\n[bold green]Selected best fold: {best_idx + 1}[/bold green] "
                         f"(val_accuracy={best_metrics.get('val_accuracy', 0.0):.1%})")
        
        return best_trainer, best_metrics
    
    elif wf_cfg.aggregate_method == "average":
        # Return last trainer but with averaged metrics
        averaged_metrics = {}
        for key in fold_metrics[0].keys():
            values = [m.get(key, 0.0) for m in fold_metrics]
            averaged_metrics[key] = np.mean(values)
            averaged_metrics[f'{key}_std'] = np.std(values)
        
        if console:
            console.print(f"\n[bold green]Averaged metrics across {len(fold_metrics)} folds[/bold green]")
        
        return fold_trainers[-1], averaged_metrics
    
    else:  # "ensemble" or unknown
        # Return last trainer (default behavior)
        if console:
            console.print(f"\n[yellow]Using last fold trainer (aggregate_method={wf_cfg.aggregate_method})[/yellow]")
        return fold_trainers[-1], fold_metrics[-1]


# — Raynergy-svg —
