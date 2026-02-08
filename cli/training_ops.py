#!/usr/bin/env python3
"""
Training operations module for ML Engine CLI.

Contains functions for training RL position sizer and retraining gate models.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    track,
)
from rich.table import Table

from cli.io_utils import DEFAULT_CONFIG_PATH
from cli.tf_config import _configure_tf_metal

console = Console()

MODEL_DIR_PATH = "trained_data/models"
MODULAR_ENSEMBLE_META_FILENAME = "modular_ensemble.meta.json"


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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    _configure_tf_metal(verbose=False)

    from src.core.modular_inference import ModularEnsembleInference
    from src.data.feature_engineering import FeatureEngineering
    from src.utils.fx_paper import candles_to_ohlcv_df

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
    import sklearn
    import xgboost

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
    from src.data.feature_engineering import FeatureEngineering

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


def train_rl_gates(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    timesteps: int = 100_000,
    pairs: str | None = None,
    granularity: str = "H1",
    candles: int = 5000,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """Train RL gate threshold optimizer using SAC.

    Learns to adapt confidence, momentum, and risk thresholds based on market regime.
    Uses a two-phase approach to avoid TensorFlow/PyTorch GPU conflicts.
    """
    import tempfile

    console.print()
    console.print(Panel.fit(
        "[bold cyan]RL Gate Threshold Training[/bold cyan]\n"
        f"[dim]SAC - {timesteps:,} timesteps - {granularity}[/dim]",
        border_style="cyan"
    ))

    modular_meta_path = Path(MODEL_DIR_PATH) / MODULAR_ENSEMBLE_META_FILENAME
    if not modular_meta_path.exists():
        console.print("[yellow]⚠ Train ensemble first: buddy train --oanda-live -n 12000[/yellow]")
        return

    ALL_MAJOR_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]
    pair_list = [p.strip() for p in pairs.split(",")] if pairs else ALL_MAJOR_PAIRS[:3]

    console.print(f"[cyan]Pairs: {', '.join(pair_list)}[/cyan]")

    # Phase 1: Generate data using TensorFlow
    console.print("\n[bold]Phase 1: Generating predictions with ensemble...[/bold]")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    _configure_tf_metal(verbose=False)

    from src.core.modular_inference import ModularEnsembleInference
    from src.data.feature_engineering import FeatureEngineering
    from src.utils.fx_paper import candles_to_ohlcv_df
    from src.utils.oanda_practice import OandaPracticeClient

    ensemble = ModularEnsembleInference(use_rl_sizer=False)
    ensemble.load_models()

    try:
        oanda = OandaPracticeClient.from_env()
    except (RuntimeError, ConnectionError, ValueError) as e:
        console.print(f"[red]OANDA connection failed: {e}[/red]")
        return

    all_prices: list[Any] = []
    all_features: list[Any] = []
    all_preds: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Fetching data...", total=len(pair_list))
        for pair in pair_list:
            try:
                resp = oanda.get_candles(pair, granularity=granularity, count=min(candles, 5000))
                df = candles_to_ohlcv_df(resp)
                fe = FeatureEngineering({})
                df = fe.create_features(df, include_all=True)
                df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

                prices = df['close'].values
                all_prices.extend(prices)

                feature_cols = [c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume']]
                features = df[feature_cols].values
                all_features.extend(features)

                for i in range(60, len(df)):
                    try:
                        signal = ensemble.predict(df.iloc[:i + 1])
                        pred = {
                            'tcn_prob': signal.tcn_probability,
                            'ridge_conf': signal.ridge_confidence,
                            'xgb_mom': signal.xgb_momentum,
                            'rf_dd': signal.rf_drawdown_pips,
                        }
                        all_preds.append(pred)
                    except (AttributeError, ValueError, KeyError):
                        all_preds.append({'tcn_prob': 0.5, 'ridge_conf': 50, 'xgb_mom': 0.5, 'rf_dd': 5})
                progress.update(task, advance=1)
            except (RuntimeError, ValueError) as e:
                console.print(f"[yellow]⚠ {pair}: {e}[/yellow]")
                progress.update(task, advance=1)

    if len(all_prices) < 200:
        console.print("[red]Insufficient data for training[/red]")
        return

    data_file = Path(tempfile.gettempdir()) / "rl_gates_data.npz"
    np.savez(
        data_file,
        prices=np.array(all_prices[:len(all_preds)]),
        features=np.array(all_features[:len(all_preds)]),
        preds=np.array([[p['tcn_prob'], p['ridge_conf'], p['xgb_mom'], p['rf_dd']] for p in all_preds]),
    )
    console.print(f"[green]✓ Generated {len(all_preds):,} samples[/green]")

    # Phase 2: Train SAC in subprocess (no TensorFlow)
    console.print("\n[bold]Phase 2: Training SAC model...[/bold]")

    cmd = [
        "python", "-c",
        f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from src.rl.gate_threshold_env import GateThresholdRL

data = np.load("{data_file}", allow_pickle=True)
prices = data["prices"]
features = data["features"]
preds = data["preds"]

print(f"Training on {{len(prices):,}} samples...")
rl = GateThresholdRL()
stats = rl.train(prices=prices, features=features, ensemble_preds=preds, timesteps={timesteps}, verbose=1)
rl.save()
print("Model saved to: trained_data/models/sac_gate_thresholds.zip")
'''
    ]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Training SAC...", total=None)
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        train_time = time.time() - start_time

    if data_file.exists():
        data_file.unlink()

    if result.returncode != 0:
        console.print(f"[red]Training failed:[/red]\n{result.stderr}")
        return

    console.print(f"[dim]{result.stdout}[/dim]")
    console.print(Panel(
        f"[bold green]✓ RL Gate Thresholds Trained[/bold green]\n\n"
        f"Timesteps: {timesteps:,}  •  Time: {train_time:.0f}s  •  Samples: {len(all_preds):,}\n\n"
        f"[bold]Usage:[/bold]  buddy --use-rl-gates -I EUR_USD",
        border_style="green"
    ))


def train_rl_exits(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    timesteps: int = 100_000,
    pairs: str | None = None,
    granularity: str = "H1",
    candles: int = 5000,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """Train RL exit timing optimizer using PPO.

    Learns optimal exit decisions (hold/take-profit/stop-loss) based on
    current PnL, time held, momentum, and volatility.
    """
    import tempfile

    console.print()
    console.print(Panel.fit(
        "[bold cyan]RL Exit Timing Training[/bold cyan]\n"
        f"[dim]PPO - {timesteps:,} timesteps - {granularity}[/dim]",
        border_style="cyan"
    ))

    modular_meta_path = Path(MODEL_DIR_PATH) / MODULAR_ENSEMBLE_META_FILENAME
    if not modular_meta_path.exists():
        console.print("[yellow]⚠ Train ensemble first: buddy train --oanda-live -n 12000[/yellow]")
        return

    ALL_MAJOR_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "USD_CAD", "NZD_USD"]
    pair_list = [p.strip() for p in pairs.split(",")] if pairs else ALL_MAJOR_PAIRS[:3]

    console.print(f"[cyan]Pairs: {', '.join(pair_list)}[/cyan]")

    # Phase 1: Generate scenarios data
    console.print("\n[bold]Phase 1: Generating trade scenarios...[/bold]")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    _configure_tf_metal(verbose=False)

    from src.data.feature_engineering import FeatureEngineering
    from src.utils.fx_paper import candles_to_ohlcv_df
    from src.utils.oanda_practice import OandaPracticeClient

    try:
        oanda = OandaPracticeClient.from_env()
    except (RuntimeError, ConnectionError, ValueError) as e:
        console.print(f"[red]OANDA connection failed: {e}[/red]")
        return

    all_scenarios: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Generating scenarios...", total=len(pair_list))
        for pair in pair_list:
            try:
                resp = oanda.get_candles(pair, granularity=granularity, count=min(candles, 5000))
                df = candles_to_ohlcv_df(resp)
                fe = FeatureEngineering({})
                df = fe.create_features(df, include_all=True)
                df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

                prices = df['close'].values
                atr = df['atr_pct_14'].values if 'atr_pct_14' in df.columns else np.full(len(df), 0.01)
                momentum = df['returns_2'].values if 'returns_2' in df.columns else np.zeros(len(df))

                for i in range(100, len(df) - 50):
                    entry_price = prices[i]
                    direction = 1 if np.random.random() > 0.5 else -1
                    for hold_bars in range(1, 25):
                        if i + hold_bars >= len(df):
                            break
                        exit_price = prices[i + hold_bars]
                        pnl_pct = direction * (exit_price - entry_price) / entry_price
                        scenario = {
                            'pnl': pnl_pct,
                            'bars_held': hold_bars,
                            'momentum': momentum[i + hold_bars] if i + hold_bars < len(momentum) else 0,
                            'atr': atr[i + hold_bars] if i + hold_bars < len(atr) else 0.01,
                            'best_exit': 1 if pnl_pct > 0.002 else (2 if pnl_pct < -0.003 else 0),
                        }
                        all_scenarios.append(scenario)
                progress.update(task, advance=1)
            except (RuntimeError, ValueError) as e:
                console.print(f"[yellow]⚠ {pair}: {e}[/yellow]")
                progress.update(task, advance=1)

    if len(all_scenarios) < 1000:
        console.print("[red]Insufficient scenarios for training[/red]")
        return

    data_file = Path(tempfile.gettempdir()) / "rl_exits_data.npz"
    np.savez(
        data_file,
        scenarios=np.array([[s['pnl'], s['bars_held'], s['momentum'], s['atr'], s['best_exit']]
                           for s in all_scenarios]),
    )
    console.print(f"[green]✓ Generated {len(all_scenarios):,} scenarios[/green]")

    # Phase 2: Train PPO in subprocess
    console.print("\n[bold]Phase 2: Training PPO model...[/bold]")

    cmd = [
        "python", "-c",
        f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.rl.optimal_exit_env import OptimalExitRL

print("Training PPO on synthetic scenarios...")
rl = OptimalExitRL()
stats = rl.train(timesteps={timesteps}, verbose=1)
rl.save()
print("Model saved to: trained_data/models/ppo_optimal_exit.zip")
'''
    ]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Training PPO...", total=None)
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        train_time = time.time() - start_time

    if data_file.exists():
        data_file.unlink()

    if result.returncode != 0:
        console.print(f"[red]Training failed:[/red]\n{result.stderr}")
        return

    console.print(f"[dim]{result.stdout}[/dim]")
    console.print(Panel(
        f"[bold green]✓ RL Exit Timing Trained[/bold green]\n\n"
        f"Timesteps: {timesteps:,}  •  Time: {train_time:.0f}s  •  Scenarios: {len(all_scenarios):,}\n\n"
        f"[bold]Usage:[/bold]  buddy --use-rl-exits -I EUR_USD",
        border_style="green"
    ))


def retrain_all(
    config_path: str = DEFAULT_CONFIG_PATH,
    *,
    granularity: str = "H1",
    candles: int = 15000,
    verbose: bool = False,
    **kwargs: Any,
) -> None:
    """Retrain ALL models for all 7 major FX pairs sequentially.

    This command trains the full 4-model ensemble (Transformer, XGBoost, RF, LightGBM)
    for each pair. Runs sequentially to avoid memory issues.
    """
    from cli.training import train_buddy as _train_buddy

    MAJOR_PAIRS = [
        "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
        "NZD_USD", "USD_CAD", "USD_CHF",
    ]

    console.print(Panel(
        f"[bold]Full Ensemble Retraining - All Pairs[/bold]\n\n"
        f"Pairs: {', '.join(MAJOR_PAIRS)}\n"
        f"Candles per pair: {candles:,}\n"
        f"Granularity: {granularity}\n\n"
        "[bold yellow]This will take 2-3 hours![/bold yellow]\n\n"
        "[dim]Each pair trains: Transformer + XGBoost + RF + LightGBM[/dim]",
        title="🔄 Retrain All Pairs",
        border_style="cyan"
    ))

    results: dict[str, dict[str, Any]] = {}
    start_time = time.time()

    for i, pair in enumerate(MAJOR_PAIRS, 1):
        pair_start = time.time()
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold cyan]Training {pair} ({i}/{len(MAJOR_PAIRS)})[/bold cyan]")
        console.print(f"{'=' * 60}\n")

        try:
            _train_buddy(
                config_path=config_path,
                instrument=pair,
                csv_path=None,
                oanda_fetch=True,
                oanda_granularity=granularity,
                oanda_candles=candles,
                **kwargs,
            )
            pair_time = time.time() - pair_start
            results[pair] = {'status': 'success', 'time_seconds': pair_time}
            console.print(f"\n[green]✓ {pair} trained in {pair_time / 60:.1f} minutes[/green]")
        except (RuntimeError, ValueError) as e:
            pair_time = time.time() - pair_start
            results[pair] = {'status': 'failed', 'error': str(e), 'time_seconds': pair_time}
            console.print(f"\n[red]✗ {pair} failed: {e}[/red]")
            continue

    total_time = time.time() - start_time
    successful = [p for p, r in results.items() if r['status'] == 'success']
    failed = [p for p, r in results.items() if r['status'] == 'failed']

    console.print("\n" + "=" * 60)
    console.print(Panel(
        f"[bold green]Retrain All Complete![/bold green]\n\n"
        f"[bold]Total time:[/bold] {total_time / 60:.1f} minutes ({total_time / 3600:.1f} hours)\n\n"
        f"[bold green]Successful ({len(successful)}):[/bold green] {', '.join(successful) or 'None'}\n"
        f"[bold red]Failed ({len(failed)}):[/bold red] {', '.join(failed) or 'None'}\n\n"
        f"[dim]Models saved to: trained_data/models/<PAIR>/[/dim]",
        title="✅ Complete",
        border_style="green"
    ))

    summary_path = Path("trained_data/retrain_all_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump({
            'completed_at': datetime.now().isoformat(),
            'total_time_seconds': total_time,
            'granularity': granularity,
            'candles': candles,
            'results': results,
        }, f, indent=2)
    console.print(f"[dim]Summary saved to: {summary_path}[/dim]")
