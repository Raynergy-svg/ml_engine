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
            import warnings
            import tensorflow as tf

            # Suppress TF mixed precision compatibility warnings (e.g. on Metal)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*mixed_float16.*")
                warnings.filterwarnings("ignore", message=".*Mixed precision.*")
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
            console.print("[dim]Mixed precision enabled: mixed_float16[/dim]")
            mp_enabled = True
        except Exception:
            console.print("[dim]Mixed precision: not available[/dim]")

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
    try:
        from oanda_practice import OandaPracticeClient
        oanda = OandaPracticeClient.from_env()
    except Exception as e:
        raise RuntimeError(f"Failed to connect to OANDA: {e}")
    
    # OANDA max is 5000 candles per request
    MAX_CANDLES_PER_REQUEST = 5000
    
    all_dfs = []
    
    for pair in pairs:
        try:
            pair_rows = []
            remaining = candles_per_pair
            to_time = None  # Start from most recent
            
            # Fetch in batches if needed
            while remaining > 0:
                batch_size = min(remaining, MAX_CANDLES_PER_REQUEST)
                
                # Build request params
                params = {
                    "granularity": granularity,
                    "count": batch_size,
                    "price": "MBA",
                }
                if to_time:
                    params["to_time"] = to_time
                
                response = oanda.get_candles(pair, **params)
                
                candles = response.get("candles", [])
                if not candles:
                    break
                
                for c in candles:
                    mid = c.get("mid", {})
                    row = {
                        "time": c.get("time"),
                        "open": float(mid.get("o", 0)),
                        "high": float(mid.get("h", 0)),
                        "low": float(mid.get("l", 0)),
                        "close": float(mid.get("c", 0)),
                        "volume": int(c.get("volume", 0)),
                    }
                    pair_rows.append(row)
                
                # Update for next batch - use oldest candle's time
                if len(candles) > 0:
                    to_time = candles[0].get("time")  # Oldest candle in this batch
                
                remaining -= len(candles)
                
                # If we got fewer candles than requested, we've hit the end
                if len(candles) < batch_size:
                    break
            
            if not pair_rows:
                if console:
                    console.print(f"  [yellow]⚠ {pair}: no candles returned[/yellow]")
                continue
            
            df = pd.DataFrame(pair_rows)
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)
            df = df.drop_duplicates(subset=["time"])  # Remove any duplicates from pagination
            
            if len(df) < 100:
                if console:
                    console.print(f"  [yellow]⚠ {pair}: insufficient data ({len(df)} rows)[/yellow]")
                continue
            
            # Add pair identifier (optional, for conditioning)
            df['pair'] = pair
            
            # Normalize OHLCV to percentage returns (instrument-agnostic)
            # This makes models generalizable across pairs
            df['open_pct'] = df['open'].pct_change()
            df['high_pct'] = (df['high'] - df['open']) / df['open']
            df['low_pct'] = (df['low'] - df['open']) / df['open']
            df['close_pct'] = df['close'].pct_change()
            
            # Normalize volume to z-score (relative to pair's own history)
            if 'volume' in df.columns:
                vol_mean = df['volume'].rolling(100, min_periods=10).mean()
                vol_std = df['volume'].rolling(100, min_periods=10).std().clip(lower=1e-8)
                df['volume_zscore'] = (df['volume'] - vol_mean) / vol_std
            
            # Drop first row with NaN from pct_change
            df = df.iloc[1:].copy()
            
            all_dfs.append(df)
            
            if console:
                console.print(f"  [green]✓ {pair}: {len(df):,} rows[/green]")
                
        except Exception as e:
            if console:
                console.print(f"  [red]✗ {pair}: {e}[/red]")
            continue
    
    if not all_dfs:
        raise ValueError("No data loaded from any pairs")
    
    # Concatenate all DataFrames
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Drop the time column to avoid duplicate index issues in feature engineering
    # Time-based features are not meaningful for shuffled multi-pair data anyway
    if 'time' in combined.columns:
        combined = combined.drop(columns=['time'])
    
    # Drop the pair column too (was for debugging, not needed for training)
    if 'pair' in combined.columns:
        combined = combined.drop(columns=['pair'])
    
    # Shuffle to mix pairs (important for training)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    if console:
        console.print(f"\n  [bold]Total: {len(combined):,} rows from {len(all_dfs)} pairs[/bold]")
    
    return combined
# — Raynergy-svg —
