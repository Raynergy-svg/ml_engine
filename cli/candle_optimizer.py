#!/usr/bin/env python3
"""Optimal candle-count finder for ML Engine Trading Bot.

Searches for the best number of historical candles to train on for a given
instrument by running lightweight walk-forward cross-validation at each
candidate count and ranking them with a composite score.

Usage (shell):
    buddy find -i EUR_USD
    buddy find -i USD_JPY --min 5000 --max 20000 --step 5000

Usage (Python):
    python main.py find-candles --instrument EUR_USD

The composite score blends:
    40%  Balanced accuracy   (class-balanced directional accuracy)
    25%  Stability           (1 - CV of accuracy across folds)
    20%  Sharpe ratio        (risk-adjusted strategy returns)
    15%  Profit factor       (gross profit / gross loss)
"""
from __future__ import annotations

import gc
import shutil
import tempfile
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from cli.config import OandaFetchOptions
from cli.io_utils import (
    DEFAULT_CONFIG_PATH,
    _normalize_instrument,
    _oanda_fetch_to_csv,
    _validate_instrument,
    console,
    logger,
)
from src.training.modular_trainers import TrainerConfig
from src.utils import load_config

__all__ = ["find_optimal_candles"]


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def _compute_composite_score(metrics: Dict[str, float]) -> float:
    """Return a 0-100 composite score from walk-forward metrics.

    Weights:
        40%  balanced accuracy  (already 0-1)
        25%  stability          max(0, 1 - CV)  where CV = std/mean
        20%  sharpe ratio       clip(sharpe / 2, 0, 1)
        15%  profit factor      clip((pf - 1) / 2, 0, 1)
    """
    bal_acc = float(metrics.get("mean_balanced_accuracy", 0.0))
    stability_cv = float(metrics.get("stability_score", 1.0))
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    pf = float(metrics.get("profit_factor", 1.0))

    bal_score = np.clip(bal_acc, 0.0, 1.0)
    stab_score = np.clip(1.0 - stability_cv, 0.0, 1.0)
    sharpe_score = np.clip(sharpe / 2.0, 0.0, 1.0)
    pf_score = np.clip((pf - 1.0) / 2.0, 0.0, 1.0)

    composite = (
        0.40 * bal_score
        + 0.25 * stab_score
        + 0.20 * sharpe_score
        + 0.15 * pf_score
    )
    return round(float(composite) * 100, 2)


# ---------------------------------------------------------------------------
# Per-count evaluation
# ---------------------------------------------------------------------------

def _make_sequences(x_flat: np.ndarray, y_flat: np.ndarray, w_flat: Optional[np.ndarray], sl: int):
    """Create (samples, seq_len, features) sequences from flat arrays."""
    n = len(x_flat)
    if n <= sl:
        return None, None, None
    seqs, labs, wts = [], [], []
    for i in range(sl, n):
        seqs.append(x_flat[i - sl : i])
        labs.append(y_flat[i])
        if w_flat is not None:
            wts.append(w_flat[i])
    return np.array(seqs), np.array(labs), np.array(wts) if wts else None


def _fetch_and_prepare_data(
    config_path: str,
    instrument: str,
    granularity: str,
    candle_count: int,
) -> Optional[tuple[pd.DataFrame, Dict[str, Any], int, List[str]]]:
    """Fetch data and compute features. Returns (df, data, n_clear, feature_names) or None."""
    from src.core.modular_data_loaders import compute_normalized_features, load_direction_data

    cfg = load_config(config_path)

    # --- fetch data --------------------------------------------------------
    try:
        opts = OandaFetchOptions(
            instrument=instrument,
            granularity=granularity,
            candles=candle_count,
            price="MBA",
        )
        csv_path = _oanda_fetch_to_csv(opts)
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Fetch failed for %s candles: %s", candle_count, exc)
        return None

    if df is None or len(df) < 500:
        logger.warning("Insufficient data for %s candles (%d rows)", candle_count, len(df) if df is not None else 0)
        return None

    # --- features ----------------------------------------------------------
    try:
        if "returns_1" not in df.columns:
            df = compute_normalized_features(df)

        direction_cfg = cfg.get("direction", {})
        lookahead = int(direction_cfg.get("lookahead", cfg.get("direction_lookahead", 24)))
        threshold = float(direction_cfg.get("threshold", cfg.get("direction_threshold", 0.003)))

        data = load_direction_data(df, lookahead=lookahead, threshold=threshold)
    except Exception as exc:
        logger.warning("Feature prep failed for %s candles: %s", candle_count, exc)
        return None

    x_train = data["X_train"]
    y_train = data["y_train"]
    w_train = data.get("w_train")
    x_val = data["X_val"]
    y_val = data["y_val"]

    # We need enough clear labels for walk-forward folds
    clear_mask_train = ~np.isclose(y_train, 0.5) if w_train is None else (w_train > 0)
    n_clear = int(np.sum(clear_mask_train))
    feature_names = data.get("feature_names", [])

    if n_clear < 300:
        logger.warning("Only %d clear samples for %s candles — skipping", n_clear, candle_count)
        return None

    # Combine train+val for walk-forward splitting
    x_all = np.concatenate([x_train, x_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    w_all = np.concatenate([w_train, data.get("w_val", np.ones(len(y_val)))], axis=0) if w_train is not None else None
    
    data["X_all"] = x_all
    data["y_all"] = y_all
    data["w_all"] = w_all
    
    return df, data, n_clear, feature_names


def _process_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    x_all: np.ndarray,
    y_all: np.ndarray,
    w_all: Optional[np.ndarray],
    df: pd.DataFrame,
    feature_names: List[str],
    trainer_cfg: TrainerConfig,
    effective_seq_len: int,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Process a single fold. Returns (bal_acc, sharpe, pf) or (None, None, None)."""
    import tensorflow as tf
    from src.training.modular_trainers import TransformerDirectionTrainer
    from src.training.walkforward_validation import calculate_trading_metrics

    # Pass raw 2D data — the trainer handles scaling + sequencing internally
    x_train = x_all[train_idx]
    y_train = y_all[train_idx]
    w_train = w_all[train_idx] if w_all is not None else None
    x_val = x_all[val_idx]
    y_val = y_all[val_idx]
    w_val = w_all[val_idx] if w_all is not None else None

    # Need enough raw samples for the trainer to create sequences
    if len(x_train) < effective_seq_len + 100 or len(x_val) < effective_seq_len + 30:
        return None, None, None

    try:
        trainer = TransformerDirectionTrainer(trainer_cfg)
        metrics = trainer.train(x_train, y_train, x_val, y_val, feature_names=feature_names, w_train=w_train, w_val=w_val)
        bal_acc = float(metrics.get("val_balanced_accuracy", metrics.get("val_accuracy", 0.0)))

        # Trading metrics from validation predictions
        try:
            # Build val sequences the same way the trainer does for prediction
            from sklearn.preprocessing import StandardScaler
            scaler = trainer.scaler if hasattr(trainer, 'scaler') and trainer.scaler is not None else StandardScaler().fit(x_train)
            x_val_scaled = scaler.transform(x_val)
            seq_len = trainer.seq_len if hasattr(trainer, 'seq_len') else effective_seq_len
            x_val_seqs = []
            y_val_seqs = []
            for i in range(seq_len, len(x_val_scaled)):
                x_val_seqs.append(x_val_scaled[i - seq_len:i])
                y_val_seqs.append(y_val[i])
            if len(x_val_seqs) < 2:
                raise ValueError("Not enough val sequences")
            x_val_seqs = np.array(x_val_seqs)
            y_val_seqs = np.array(y_val_seqs)

            val_preds = trainer.model.predict(x_val_seqs, verbose=0).flatten()
            close_prices = df["close"].values
            val_start = val_idx[0] + seq_len
            val_end = min(val_start + len(val_preds), len(close_prices))
            price_slice = close_prices[val_start:val_end]

            n_common = min(len(val_preds), len(price_slice))
            val_preds = val_preds[:n_common]
            price_slice = price_slice[:n_common]
            y_val_seqs = y_val_seqs[:n_common]

            if n_common > 1:
                trading = calculate_trading_metrics(y_val_seqs, val_preds, price_slice)
                sharpe = float(trading.get("sharpe_ratio", 0.0))
                pf = float(trading.get("profit_factor", 1.0))
            else:
                sharpe, pf = 0.0, 1.0
        except Exception:
            sharpe, pf = 0.0, 1.0

        return bal_acc, sharpe, pf
    except Exception as exc:
        logger.warning("Fold %d processing failed: %s", fold_idx, exc, exc_info=True)
        return None, None, None
    finally:
        tf.keras.backend.clear_session()
        gc.collect()


def _evaluate_candle_count(
    config_path: str,
    instrument: str,
    granularity: str,
    candle_count: int,
    n_folds: int = 3,
    seq_len: int = 60,
    *,
    progress_cb=None,
) -> Optional[Dict[str, Any]]:
    """Fetch *candle_count* candles and run lightweight walk-forward CV.

    Returns a metrics dict or ``None`` if the evaluation failed.
    """
    from src.training.walkforward_validation import WalkForwardValidator

    # --- fetch and prepare -----------------------------------------------
    prep_result = _fetch_and_prepare_data(config_path, instrument, granularity, candle_count)
    if prep_result is None:
        return None

    df, data, n_clear, feature_names = prep_result
    x_all = data["X_all"]
    y_all = data["y_all"]
    w_all = data["w_all"]
    n_features = x_all.shape[1]

    # --- validate sequence length ------------------------------------------
    effective_seq_len = min(seq_len, len(x_all) // 4)
    if effective_seq_len < 10:
        logger.warning("Sequence length too short for %s candles — skipping", candle_count)
        return None

    # --- create validator --------------------------------------------------
    # Sizes must be small enough for n_folds expanding windows to fit:
    # train_size*N + (n_folds-1)*fold_size + gap + val + test <= N
    validator = WalkForwardValidator(
        n_splits=n_folds,
        train_size=0.50,
        val_size=0.08,
        test_size=0.05,
        gap=max(10, effective_seq_len // 2),
        mode="expanding",
        min_train_size=200,
    )

    # --- setup trainer config ----------------------------------------------
    trainer_cfg = TrainerConfig(
        epochs=15,
        batch_size=64,
        learning_rate=0.0005,
        patience=5,
        min_epochs=5,
        verbose=0,
        transformer_d_model=32,
        transformer_num_heads=4,
        transformer_num_layers=2,
        transformer_dropout=0.2,
        use_focal_loss=True,
        use_ema=False,
        use_ewc=False,
        use_replay_buffer=False,
    )

    tmp_dir = tempfile.mkdtemp(prefix="candle_opt_")
    trainer_cfg = replace(trainer_cfg, checkpoint_dir=tmp_dir)

    # --- walk-forward CV ---------------------------------------------------
    fold_bal_accs: List[float] = []
    fold_sharpes: List[float] = []
    fold_pfs: List[float] = []

    try:
        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(validator.split(x_all)):
            if progress_cb:
                progress_cb(fold_idx + 1, n_folds)

            bal_acc, sharpe, pf = _process_fold(
                fold_idx, train_idx, val_idx, x_all, y_all, w_all,
                df, feature_names, trainer_cfg, effective_seq_len,
            )
            
            if bal_acc is not None:
                fold_bal_accs.append(bal_acc)
                fold_sharpes.append(sharpe)
                fold_pfs.append(pf)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- aggregate results -------------------------------------------------
    if len(fold_bal_accs) < 1:
        logger.warning("No successful folds for %s candles", candle_count)
        return None

    mean_bal = float(np.mean(fold_bal_accs))
    std_bal = float(np.std(fold_bal_accs))
    stability = std_bal / mean_bal if mean_bal > 0 else 999.0

    result: Dict[str, Any] = {
        "candles": candle_count,
        "rows_fetched": len(df),
        "clear_samples": n_clear,
        "n_features": n_features,
        "folds_completed": len(fold_bal_accs),
        "mean_balanced_accuracy": mean_bal,
        "std_balanced_accuracy": std_bal,
        "stability_score": stability,
        "sharpe_ratio": float(np.mean(fold_sharpes)) if fold_sharpes else 0.0,
        "profit_factor": float(np.mean(fold_pfs)) if fold_pfs else 1.0,
    }
    result["composite_score"] = _compute_composite_score(result)
    return result


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _display_results(
    results: List[Dict[str, Any]],
    instrument: str,
    granularity: str,
) -> Optional[Dict[str, Any]]:
    """Print a Rich table of results and return the best candidate."""
    if not results:
        console.print("[red]No valid results to display.[/red]")
        return None

    # Sort by composite score descending
    ranked = sorted(results, key=lambda r: r.get("composite_score", 0), reverse=True)
    best = ranked[0]

    table = Table(
        title=f"Candle Optimization Results — {instrument} ({granularity})",
        show_lines=True,
    )
    table.add_column("Rank", justify="center", style="dim", width=5)
    table.add_column("Candles", justify="right", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Clear Samples", justify="right")
    table.add_column("Bal. Acc", justify="right")
    table.add_column("Stability", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Profit Factor", justify="right")
    table.add_column("Score", justify="right", style="bold")

    for rank, r in enumerate(ranked, 1):
        is_best = r is best
        score_str = f"[bold green]{r['composite_score']:.1f} ★[/bold green]" if is_best else f"{r['composite_score']:.1f}"
        candle_str = f"[bold green]{r['candles']:,}[/bold green]" if is_best else f"{r['candles']:,}"
        stab_val = r.get("stability_score", 999)
        if stab_val < 0.10:
            stab_color = "green"
        elif stab_val < 0.20:
            stab_color = "yellow"
        else:
            stab_color = "red"
        stab_str = f"[{stab_color}]{stab_val:.3f}[/{stab_color}]"

        table.add_row(
            str(rank),
            candle_str,
            f"{r.get('rows_fetched', 0):,}",
            f"{r.get('clear_samples', 0):,}",
            f"{r['mean_balanced_accuracy']:.2%}",
            stab_str,
            f"{r.get('sharpe_ratio', 0):.3f}",
            f"{r.get('profit_factor', 1):.3f}",
            score_str,
        )

    console.print()
    console.print(table)
    console.print()

    # Summary panel
    console.print(Panel(
        f"[bold green]Optimal: {best['candles']:,} candles[/bold green]  "
        f"(score {best['composite_score']:.1f}/100)\n\n"
        f"[dim]Balanced Accuracy:[/dim]  {best['mean_balanced_accuracy']:.2%} ± {best['std_balanced_accuracy']:.2%}\n"
        f"[dim]Stability (CV):[/dim]     {best['stability_score']:.4f}\n"
        f"[dim]Sharpe Ratio:[/dim]       {best['sharpe_ratio']:.3f}\n"
        f"[dim]Profit Factor:[/dim]      {best['profit_factor']:.3f}\n"
        f"[dim]Clear Samples:[/dim]      {best['clear_samples']:,}",
        title=f"★  Best Configuration for {instrument}",
        border_style="green",
    ))

    return best


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_optimal_candles(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    instrument: str,
    granularity: str = "H1",
    min_candles: int = 3000,
    max_candles: int = 30000,
    step: int = 3000,
    n_folds: int = 3,
    auto_train: bool = True,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Search for the optimal training candle count for *instrument*.

    Iterates over candidate candle counts from *min_candles* to *max_candles*
    (inclusive) in steps of *step*.  For each count a lightweight 3-fold
    walk-forward cross-validation is executed.  Results are ranked by a
    composite score.

    If *auto_train* is ``True`` (default), the full ``train_buddy`` pipeline
    is launched automatically with the winning candle count.

    Returns:
        The best result dict, or ``None`` if all evaluations failed.
    """
    instrument = _normalize_instrument(instrument)
    _validate_instrument(instrument)

    cfg = load_config(config_path)
    seq_len = int(cfg.get("buddy", {}).get("train_defaults", {}).get("seq_len", 60))

    candidates = list(range(min_candles, max_candles + 1, step))
    if not candidates:
        console.print("[red]No candidate candle counts. Check --min-candles / --max-candles / --step.[/red]")
        return None

    console.print(Panel(
        f"[bold]Instrument:[/bold]    {instrument}\n"
        f"[bold]Granularity:[/bold]   {granularity}\n"
        f"[bold]Candidates:[/bold]    {', '.join(f'{c:,}' for c in candidates)}\n"
        f"[bold]CV Folds:[/bold]      {n_folds}\n"
        f"[bold]Seq Length:[/bold]    {seq_len}\n"
        f"[bold]Auto-Train:[/bold]    {'Yes' if auto_train else 'No'}",
        title="Candle Optimizer",
        border_style="cyan",
    ))

    results: List[Dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task(
            "Testing candle counts…",
            total=len(candidates),
        )

        for idx, count in enumerate(candidates):
            progress.update(
                main_task,
                description=f"[cyan]{count:,}[/cyan] candles — fetching & evaluating…",
            )

            def _fold_cb(fold: int, total: int, count: int = count):
                progress.update(
                    main_task,
                    description=f"[cyan]{count:,}[/cyan] candles — fold {fold}/{total}",
                )

            t0 = time.perf_counter()
            result = _evaluate_candle_count(
                config_path=config_path,
                instrument=instrument,
                granularity=granularity,
                candle_count=count,
                n_folds=n_folds,
                seq_len=seq_len,
                progress_cb=_fold_cb,
            )
            elapsed = time.perf_counter() - t0

            if result is not None:
                result["elapsed_sec"] = round(elapsed, 1)
                results.append(result)
                console.print(
                    f"  [green]✓[/green] {count:,} candles — "
                    f"score {result['composite_score']:.1f}  "
                    f"bal_acc {result['mean_balanced_accuracy']:.2%}  "
                    f"({elapsed:.0f}s)"
                )
            else:
                console.print(f"  [yellow]✗[/yellow] {count:,} candles — skipped (insufficient data or failed)")

            progress.advance(main_task)

            # Small pause to be friendly to OANDA rate limits
            if idx < len(candidates) - 1:
                time.sleep(1.0)

    # --- Display results ---------------------------------------------------
    best = _display_results(results, instrument, granularity)

    if best is None:
        console.print("[red]All candidate counts failed. Try a different range or check your OANDA credentials.[/red]")
        return None

    # --- Auto-train --------------------------------------------------------
    if auto_train:
        console.print()
        console.print(
            f"[bold cyan]Auto-training with optimal {best['candles']:,} candles…[/bold cyan]"
        )
        console.print()

        try:
            from types import SimpleNamespace
            from cli.commands import _dispatch_train_buddy
            from cli.training import train_buddy

            # Build a minimal args namespace that _dispatch_train_buddy expects
            train_args_dict: Dict[str, Any] = {
                "command": "train-buddy",
                "config": config_path,
                "instrument": instrument,
                "granularity": granularity,
                "candles": best["candles"],
                "oanda_live": True,
                "verbose": verbose,
                # Let defaults fill the rest
                "csv": None,
                "model_type": None,
                "seq_len": None,
                "epochs": None,
                "batch_size": None,
                "lr": None,
                "patience": None,
                "enterprise": False,
                "cv_folds": None,
                "bootstrap": False,
                "bootstrap_samples": None,
                "multi_pair": False,
                "foundation_pairs": None,
                "warm_start": False,
                "no_warm_start": False,
                "init_from": None,
                "es_monitor": None,
                "combined_w_dir": None,
                "combined_w_conf": None,
                "top_features": None,
                "feature_curriculum": None,
                "curriculum_ks": None,
                "pca_components": None,
                "train_smoothing": None,
                "median_window": None,
                "vol_norm_window": None,
                "min_volume": None,
                "spread_filter": False,
                "spread_pctl": None,
                "spread_mult": None,
                "no_mixed_precision": False,
                "mixed_precision": None,
                "workers": None,
                "train_volatility_regime": False,
                "no_volatility_regime": False,
                "tier2_stop_loss_pips": None,
                "tier2_take_profit_pips": None,
                "tier2_spread_fallback_pips": None,
                "tier2_horizon_candles": None,
                "tier2_tie_break": None,
                "tier2_calibration_bins": None,
                "tier2_calibration_stride": None,
                "seed": 42,
                "run_tag": None,
            }

            train_args = SimpleNamespace(**train_args_dict)
            _dispatch_train_buddy(train_args, {"train-buddy": train_buddy})
        except Exception as exc:
            console.print(f"[red]Auto-train failed:[/red] {exc}")
            console.print(
                f"[dim]You can train manually:[/dim]  "
                f"buddy train -i {instrument} -c {best['candles']}"
            )

    return best
