#!/usr/bin/env python3
"""Model management commands for ML Engine.

This module provides commands for managing model lifecycle:
- model_status: Display current model status and training metrics
- promote_model: Promote candidate model to production
"""

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cli.io_utils import DEFAULT_CONFIG_PATH

console = Console()


def model_status(config_path: str = DEFAULT_CONFIG_PATH, **kwargs: Any) -> None:
    """Display current model status and training metrics."""
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
