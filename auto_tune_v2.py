#!/usr/bin/env python3
"""Enhanced hyperparameter tuning v2 - explores more parameters and class imbalance fixes.

Improvements over v1:
- Explores sequence_length, weight_decay, state_weight
- Targeted search around best found config (hidden=64, layers=3, dropout=0.2)
- Supports class-weighted loss for imbalanced state labels
- Multiple search strategies: grid, random, targeted

Usage:
    python auto_tune_v2.py --config config.yaml --instrument EUR_USD --granularity M5 --candles 1500 --strategy targeted
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ─────────────────────────────────────────────────────────────────────────────
# SEARCH GRIDS
# ─────────────────────────────────────────────────────────────────────────────

# Full exploration grid (162 combinations)
FULL_GRID = {
    "hidden_size": [32, 64, 128],
    "num_layers": [2, 3, 4],
    "dropout": [0.15, 0.2, 0.25, 0.3],
    "learning_rate": [3e-4, 5e-4, 8e-4],
    "batch_size": [64],
    "weight_decay": [0.0, 1e-4],
    "sequence_length": [60],
}

# Targeted search around winner (hidden=64, layers=3, dropout=0.2) - 36 combos
TARGETED_GRID = {
    "hidden_size": [48, 64, 80],          # Explore around 64
    "num_layers": [3, 4],                  # 3 was best, try 4
    "dropout": [0.15, 0.2, 0.25],          # Explore around 0.2
    "learning_rate": [3e-4, 5e-4, 7e-4],   # Fine-tune LR
    "batch_size": [64],
    "weight_decay": [0.0, 5e-5],           # Light regularization
    "state_weight": [5.0, 7.0],            # Boost state head
}

# Quick sanity check grid (8 combos)
QUICK_GRID = {
    "hidden_size": [64],
    "num_layers": [3, 4],
    "dropout": [0.2, 0.25],
    "learning_rate": [5e-4],
    "batch_size": [64],
    "weight_decay": [0.0, 1e-4],
    "state_weight": [5.0],
}

# Sequence length exploration (12 combos)
SEQ_GRID = {
    "hidden_size": [64],
    "num_layers": [3],
    "dropout": [0.2],
    "learning_rate": [5e-4],
    "batch_size": [64],
    "weight_decay": [0.0],
    "sequence_length": [30, 45, 60, 90, 120, 150],
    "state_weight": [5.0, 7.0],
}


STRATEGY_GRIDS = {
    "full": FULL_GRID,
    "targeted": TARGETED_GRID,
    "quick": QUICK_GRID,
    "sequence": SEQ_GRID,
}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def apply_params(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply hyperparameters to config with all new params."""
    cfg = copy.deepcopy(config)
    
    # Model architecture params
    if "model" not in cfg:
        cfg["model"] = {}
    
    cfg["model"]["hidden_size"] = params.get("hidden_size", cfg["model"].get("hidden_size", 64))
    cfg["model"]["num_layers"] = params.get("num_layers", cfg["model"].get("num_layers", 3))
    cfg["model"]["dropout"] = params.get("dropout", cfg["model"].get("dropout", 0.2))
    
    # Root level for compatibility
    cfg["hidden_size"] = cfg["model"]["hidden_size"]
    cfg["num_layers"] = cfg["model"]["num_layers"]
    cfg["dropout"] = cfg["model"]["dropout"]
    
    # Training params
    cfg["learning_rate"] = params.get("learning_rate", cfg.get("learning_rate", 5e-4))
    cfg["batch_size"] = params.get("batch_size", cfg.get("batch_size", 64))
    cfg["weight_decay"] = params.get("weight_decay", cfg.get("weight_decay", 0.0))
    
    # Sequence length (data windowing)
    if "sequence_length" in params:
        cfg["sequence_length"] = params["sequence_length"]
        if "data" in cfg:
            cfg["data"]["sequence_length"] = params["sequence_length"]
    
    # State head loss weight
    if "state_weight" in params:
        if "unified_head_loss_weights" not in cfg:
            cfg["unified_head_loss_weights"] = {"price": 1.0, "trend": 0.5, "risk": 0.5, "state": 5.0}
        cfg["unified_head_loss_weights"]["state"] = params["state_weight"]
    
    # Class weights for imbalanced state labels (calculated from data: {0: 670, 1: 626})
    # Weight = total / (num_classes * class_count) → balanced emphasis
    if params.get("use_class_weights", False):
        cfg["state_class_weights"] = [0.967, 1.035]  # Slight boost to minority class
    
    # Tuning-specific: reduce epochs for faster iteration
    cfg["epochs"] = params.get("epochs", 30)
    cfg["min_epochs"] = min(10, cfg["epochs"])
    cfg["early_stop_patience"] = 5
    
    return cfg


def run_training(
    config_path: str,
    instrument: str,
    granularity: str,
    candles: int,
) -> Dict[str, Any]:
    """Run unified training and return metrics."""
    from oanda_unified_training import run_oanda_unified_training_session
    
    try:
        result = run_oanda_unified_training_session(
            config_path,
            instruments=instrument,
            granularity=granularity,
            candles=candles,
            resume_path=None,
        )
        
        metrics_path = result.metrics_path
        if metrics_path and Path(metrics_path).exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            return {"success": True, "metrics": metrics, "model_path": result.model_path}
        else:
            return {"success": True, "metrics": {}, "model_path": result.model_path}
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def extract_best_score(result: Dict[str, Any]) -> Tuple[Optional[float], Optional[Dict]]:
    """Extract best validation state accuracy and full metrics."""
    if not result.get("success"):
        return None, None
    
    metrics = result.get("metrics", {})
    history = metrics.get("history", [])
    
    if not history:
        return None, None
    
    best_acc = 0.0
    best_entry = None
    for entry in history:
        val = entry.get("val", {})
        acc = val.get("state_acc", 0.0)
        if acc > best_acc:
            best_acc = acc
            best_entry = entry
    
    return (best_acc, best_entry) if best_acc > 0 else (None, None)


def generate_param_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters."""
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


def generate_random_sample(grid: Dict[str, List[Any]], n: int = 20) -> List[Dict[str, Any]]:
    """Generate random sample from grid (for large search spaces)."""
    combinations = generate_param_combinations(grid)
    if len(combinations) <= n:
        return combinations
    return random.sample(combinations, n)


def run_tuning(
    base_config_path: str,
    instrument: str,
    granularity: str,
    candles: int,
    strategy: str = "targeted",
    max_combos: Optional[int] = None,
) -> Dict[str, Any]:
    """Run hyperparameter tuning with specified strategy."""
    
    grid = STRATEGY_GRIDS.get(strategy, TARGETED_GRID)
    combinations = generate_param_combinations(grid)
    
    # Limit combinations if specified
    if max_combos and len(combinations) > max_combos:
        print(f"Sampling {max_combos} from {len(combinations)} combinations")
        combinations = random.sample(combinations, max_combos)
    
    print("=" * 60)
    print(f"HYPERPARAMETER TUNING v2 - Strategy: {strategy.upper()}")
    print("=" * 60)
    print(f"Total combinations: {len(combinations)}")
    print(f"Grid keys: {list(grid.keys())}")
    print("=" * 60)
    
    base_config = load_config(base_config_path)
    
    # Create tuning results directory
    tune_dir = Path("trained_data/tuning")
    tune_dir.mkdir(parents=True, exist_ok=True)
    
    temp_config_path = str(tune_dir / "tune_config_v2.yaml")
    
    results: List[Dict[str, Any]] = []
    best_score = 0.0
    best_params: Optional[Dict[str, Any]] = None
    best_model_path: Optional[str] = None
    best_metrics: Optional[Dict] = None
    
    start_time = datetime.now()
    
    for i, params in enumerate(combinations, 1):
        iter_start = datetime.now()
        
        # Compact param display
        param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k not in ["batch_size", "epochs"])
        print(f"\n[{i}/{len(combinations)}] {param_str}")
        
        cfg = apply_params(base_config, params)
        save_config(cfg, temp_config_path)
        
        result = run_training(temp_config_path, instrument, granularity, candles)
        score, entry_metrics = extract_best_score(result)
        
        iter_time = (datetime.now() - iter_start).total_seconds()
        
        entry = {
            "params": params,
            "score": score,
            "success": result.get("success", False),
            "error": result.get("error"),
            "time_seconds": iter_time,
            "best_epoch_metrics": entry_metrics,
        }
        results.append(entry)
        
        if score is not None:
            marker = ""
            if score > best_score:
                best_score = score
                best_params = params
                best_model_path = result.get("model_path")
                best_metrics = entry_metrics
                marker = " ★ NEW BEST"
            print(f"    → state_acc: {score:.2%}{marker} ({iter_time:.0f}s)")
        else:
            print(f"    → FAILED: {result.get('error', 'unknown')[:60]}")
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Summary
    print(f"\n{'=' * 60}")
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"Best state_acc: {best_score:.2%}")
    print(f"Best params: {best_params}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    # Rank top 5
    ranked = sorted([r for r in results if r["score"]], key=lambda x: x["score"], reverse=True)
    if ranked:
        print(f"\nTop 5 configurations:")
        for rank, r in enumerate(ranked[:5], 1):
            p = r["params"]
            print(f"  {rank}. {r['score']:.2%} | h={p.get('hidden_size')}, L={p.get('num_layers')}, d={p.get('dropout')}, lr={p.get('learning_rate')}")
    
    # Save results
    results_path = tune_dir / f"tuning_v2_{strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump({
            "strategy": strategy,
            "best_score": best_score,
            "best_params": best_params,
            "best_model_path": best_model_path,
            "total_time_seconds": total_time,
            "all_results": results,
            "ranked_top10": [{"score": r["score"], "params": r["params"]} for r in ranked[:10]],
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Update main config with best params
    if best_params and best_score > 0.79:  # Only update if we beat 79%
        print(f"\n✓ Updating {base_config_path} with best params (beat 79% threshold)...")
        final_cfg = apply_params(base_config, best_params)
        final_cfg["epochs"] = base_config.get("epochs", 200)
        final_cfg["min_epochs"] = base_config.get("min_epochs", 50)
        final_cfg["early_stop_patience"] = base_config.get("early_stop_patience", 15)
        save_config(final_cfg, base_config_path)
        print("✓ Config updated")
    elif best_params:
        print(f"\n⚠ Best score {best_score:.2%} did not beat 79% threshold - config NOT updated")
        print("  Run with --force-update to update anyway")
    
    return {
        "best_score": best_score,
        "best_params": best_params,
        "best_model_path": best_model_path,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced hyperparameter tuning v2")
    parser.add_argument("--config", "-c", default="config.yaml", help="Base config file")
    parser.add_argument("--instrument", "-I", default="EUR_USD", help="OANDA instrument")
    parser.add_argument("--granularity", "-g", default="M5", help="Candle granularity")
    parser.add_argument("--candles", "-n", type=int, default=1500, help="Number of candles")
    parser.add_argument("--strategy", "-s", default="targeted", 
                       choices=["full", "targeted", "quick", "sequence"],
                       help="Search strategy: full (162), targeted (36), quick (8), sequence (12)")
    parser.add_argument("--max-combos", "-m", type=int, help="Max combinations to try (random sample)")
    parser.add_argument("--force-update", action="store_true", help="Update config even if no improvement")
    
    args = parser.parse_args()
    
    print(f"Strategy: {args.strategy}")
    print(f"Available strategies: full ({len(generate_param_combinations(FULL_GRID))}), "
          f"targeted ({len(generate_param_combinations(TARGETED_GRID))}), "
          f"quick ({len(generate_param_combinations(QUICK_GRID))}), "
          f"sequence ({len(generate_param_combinations(SEQ_GRID))})")
    
    run_tuning(
        base_config_path=args.config,
        instrument=args.instrument,
        granularity=args.granularity,
        candles=args.candles,
        strategy=args.strategy,
        max_combos=args.max_combos,
    )


if __name__ == "__main__":
    main()
