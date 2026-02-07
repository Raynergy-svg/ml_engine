#!/usr/bin/env python3
"""
Retrain ONLY the gate models (XGBoost, RF, Ridge) locally.

The Transformer direction model (78% accuracy) works fine.
The sklearn gate models have version mismatch issues and need local retraining.

Usage:
    python retrain_gates.py --pairs EUR_USD,GBP_USD --candles 5000
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Retrain gate models locally")
    parser.add_argument("--pairs", type=str, default="EUR_USD,GBP_USD,USD_JPY",
                        help="Comma-separated list of pairs to train on")
    parser.add_argument("--candles", type=int, default=5000,
                        help="Number of candles per pair")
    parser.add_argument("--granularity", type=str, default="H1",
                        help="Timeframe")
    args = parser.parse_args()
    
    pairs = [p.strip() for p in args.pairs.split(",")]
    
    console.print(Panel(
        f"[bold]Retraining Gate Models Locally[/bold]\n\n"
        f"Pairs: {', '.join(pairs)}\n"
        f"Candles per pair: {args.candles:,}\n"
        f"Granularity: {args.granularity}\n\n"
        "[dim]This will retrain XGBoost, RF, and Ridge models[/dim]\n"
        "[dim]The Transformer direction model will be kept unchanged[/dim]",
        title="🔧 Gate Model Retraining",
        border_style="cyan"
    ))
    
    # Import after arg parsing for faster --help
    from src.core.modular_data_loaders import (
        compute_normalized_features,
        load_xgboost_data,
        load_rf_data,
        load_ridge_data
    )
    from src.training.modular_trainers import (
        XGBoostTrainer,
        RandomForestTrainer,
        RidgeTrainer,
        TrainerConfig
    )
    from src.data.feature_engineering import FeatureEngineering
    
    # Fetch data from OANDA
    console.print("\n[bold]Step 1: Fetching data from OANDA...[/bold]")
    
    try:
        from oanda_practice import OandaPracticeClient
        oanda = OandaPracticeClient.from_env()
    except Exception as e:
        console.print(f"[red]Failed to connect to OANDA: {e}[/red]")
        console.print("[yellow]Using local cached data if available...[/yellow]")
        oanda = None
    
    all_dfs = []
    
    for pair in track(pairs, description="Fetching pairs..."):
        try:
            if oanda:
                response = oanda.get_candles(
                    pair,
                    granularity=args.granularity,
                    count=args.candles
                )
                # Parse OANDA response format
                candles_raw = response.get('candles', [])
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
                df = pd.DataFrame(rows)
            else:
                # Try local cache
                cache_path = Path(f"trained_data/cache/{pair}_{args.granularity}.pkl")
                if cache_path.exists():
                    with open(cache_path, 'rb') as f:
                        df = pickle.load(f)
                else:
                    console.print(f"[yellow]Skipping {pair} - no data available[/yellow]")
                    continue
            
            # Add pair identifier
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
    import sklearn
    meta['gate_models_retrained_at'] = datetime.now().isoformat()
    meta['gate_models_trained_on'] = 'local_m1'
    meta['sklearn_version'] = sklearn.__version__
    meta['retrained_pairs'] = pairs
    meta['retrained_candles'] = args.candles
    
    # Update results
    if 'results' not in meta:
        meta['results'] = {}
    meta['results']['xgboost'] = xgb_metrics
    meta['results']['random_forest'] = rf_metrics
    meta['results']['ridge'] = ridge_metrics
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    console.print("  [green]✓ Metadata updated[/green]")
    
    # Summary
    console.print("\n" + "="*60)
    console.print(Panel(
        f"[bold green]Gate Models Retrained Successfully![/bold green]\n\n"
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


if __name__ == "__main__":
    main()
