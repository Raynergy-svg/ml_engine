#!/usr/bin/env python3
"""Automated hyperparameter tuning for unified multitask training.

This script runs multiple training sessions with different hyperparameter combinations,
logs results, and saves the best configuration found.

Usage:
    python auto_tune.py --config config.yaml --instrument EUR_USD --granularity M5 --candles 1500
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Define hyperparameter search space
PARAM_GRID = {
    "hidden_size": [64, 128, 256],
    "num_layers": [2, 3],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "batch_size": [32, 64, 128],
}

# Reduced grid for quick tuning (comment out above and use this for faster runs)
QUICK_GRID = {
    "hidden_size": [64, 128],
    "num_layers": [2, 3],
    "dropout": [0.1, 0.2],
    "learning_rate": [5e-4],
    "batch_size": [64],
}


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], config_path: str) -> None:
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def apply_params(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply hyperparameters to config."""
    cfg = copy.deepcopy(config)
    
    # Model params
    if "model" not in cfg:
        cfg["model"] = {}
    cfg["model"]["hidden_size"] = params.get("hidden_size", cfg["model"].get("hidden_size", 64))
    cfg["model"]["num_layers"] = params.get("num_layers", cfg["model"].get("num_layers", 2))
    cfg["model"]["dropout"] = params.get("dropout", cfg["model"].get("dropout", 0.1))
    
    # Also set at root level for compatibility
    cfg["hidden_size"] = params.get("hidden_size", 64)
    cfg["num_layers"] = params.get("num_layers", 2)
    cfg["dropout"] = params.get("dropout", 0.1)
    
    # Training params
    cfg["learning_rate"] = params.get("learning_rate", cfg.get("learning_rate", 5e-4))
    cfg["batch_size"] = params.get("batch_size", cfg.get("batch_size", 64))
    
    # Reduce epochs for tuning (faster iterations)
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
            resume_path=None,  # Start fresh for each tuning run
        )
        
        # Load metrics from the saved file
        metrics_path = result.metrics_path
        if metrics_path and Path(metrics_path).exists():
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            return {"success": True, "metrics": metrics, "model_path": result.model_path}
        else:
            return {"success": True, "metrics": {}, "model_path": result.model_path}
    except Exception as e:
        return {"success": False, "error": str(e)}


def extract_best_score(result: Dict[str, Any]) -> Optional[float]:
    """Extract the best validation score from training result."""
    if not result.get("success"):
        return None
    
    metrics = result.get("metrics", {})
    history = metrics.get("history", [])
    
    if not history:
        return None
    
    # Find best state accuracy from history
    best_acc = 0.0
    for entry in history:
        val = entry.get("val", {})
        acc = val.get("state_acc", 0.0)
        if acc > best_acc:
            best_acc = acc
    
    return best_acc if best_acc > 0 else None


def generate_param_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters."""
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]


def run_tuning(
    base_config_path: str,
    instrument: str,
    granularity: str,
    candles: int,
    quick: bool = False,
) -> Dict[str, Any]:
    """Run hyperparameter tuning."""
    grid = QUICK_GRID if quick else PARAM_GRID
    combinations = generate_param_combinations(grid)
    
    print("=" * 60)
    print("AUTOMATED HYPERPARAMETER TUNING")
    print("=" * 60)
    print(f"Total combinations to try: {len(combinations)}")
    print(f"Grid: {grid}")
    print("=" * 60)
    
    base_config = load_config(base_config_path)
    
    # Create tuning results directory
    tune_dir = Path("trained_data/tuning")
    tune_dir.mkdir(parents=True, exist_ok=True)
    
    # Temp config for tuning runs
    temp_config_path = str(tune_dir / "tune_config.yaml")
    
    results: List[Dict[str, Any]] = []
    best_score = 0.0
    best_params: Optional[Dict[str, Any]] = None
    best_model_path: Optional[str] = None
    
    for i, params in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] Testing: {params}")
        
        # Apply params and save temp config
        cfg = apply_params(base_config, params)
        save_config(cfg, temp_config_path)
        
        # Run training
        result = run_training(temp_config_path, instrument, granularity, candles)
        
        # Extract score
        score = extract_best_score(result)
        
        entry = {
            "params": params,
            "score": score,
            "success": result.get("success", False),
            "error": result.get("error"),
        }
        results.append(entry)
        
        if score is not None:
            print(f"    → Score (state_acc): {score:.4f}")
            if score > best_score:
                best_score = score
                best_params = params
                best_model_path = result.get("model_path")
                print("    ★ NEW BEST!")
        else:
            print(f"    → Failed or no score: {result.get('error', 'unknown')}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"Best score: {best_score:.4f}")
    print(f"Best params: {best_params}")
    print(f"Best model: {best_model_path}")
    
    # Save results
    results_path = tune_dir / f"tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump({
            "best_score": best_score,
            "best_params": best_params,
            "best_model_path": best_model_path,
            "all_results": results,
        }, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Update main config with best params if found
    if best_params:
        print(f"\nUpdating {base_config_path} with best params...")
        final_cfg = apply_params(base_config, best_params)
        # Restore original epochs (not the reduced tuning epochs)
        final_cfg["epochs"] = base_config.get("epochs", 200)
        final_cfg["min_epochs"] = base_config.get("min_epochs", 50)
        final_cfg["early_stop_patience"] = base_config.get("early_stop_patience", 15)
        save_config(final_cfg, base_config_path)
        print("✓ Config updated with best params")
    
    return {
        "best_score": best_score,
        "best_params": best_params,
        "best_model_path": best_model_path,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated hyperparameter tuning")
    parser.add_argument("--config", "-c", default="config.yaml", help="Base config file")
    parser.add_argument("--instrument", "-I", default="EUR_USD", help="OANDA instrument")
    parser.add_argument("--granularity", "-g", default="M5", help="Candle granularity")
    parser.add_argument("--candles", "-n", type=int, default=1500, help="Number of candles")
    parser.add_argument("--quick", "-q", action="store_true", help="Use reduced grid for faster tuning")
    
    args = parser.parse_args()
    
    run_tuning(
        base_config_path=args.config,
        instrument=args.instrument,
        granularity=args.granularity,
        candles=args.candles,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
