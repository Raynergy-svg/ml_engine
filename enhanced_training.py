"""Enhanced training with all improvements:
1. More data (3000+ candles)
2. Data augmentation
3. Better class balancing
4. Ensemble of models
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from data_processing import prepare_sequences
from multitask_labels import build_multitask_targets, split_time_series
from neural_engine_unified import UnifiedNeuralEngine, safe_torch_load
from utils import load_config, setup_logging

logger = setup_logging(log_file="enhanced_training.log")


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 2. DATA AUGMENTATION
# ============================================================================

def augment_sequences(x: np.ndarray, noise_std: float = 0.01, 
                      time_shift_range: int = 2) -> np.ndarray:
    """Apply data augmentation to sequences.
    
    Techniques:
    - Gaussian noise injection
    - Time shift (jitter)
    - Scaling perturbation
    """
    augmented = x.copy()
    batch_size, seq_len, features = augmented.shape
    
    # 1. Gaussian noise (50% of samples)
    noise_mask = np.random.random(batch_size) < 0.5
    noise = np.random.normal(0, noise_std, (noise_mask.sum(), seq_len, features))
    augmented[noise_mask] += noise
    
    # 2. Small scaling perturbation (30% of samples)
    scale_mask = np.random.random(batch_size) < 0.3
    scales = np.random.uniform(0.98, 1.02, (scale_mask.sum(), 1, features))
    augmented[scale_mask] *= scales
    
    # 3. Time jitter - shift sequences slightly (20% of samples)
    if time_shift_range > 0:
        jitter_mask = np.random.random(batch_size) < 0.2
        for i in np.where(jitter_mask)[0]:
            shift = np.random.randint(-time_shift_range, time_shift_range + 1)
            if shift != 0:
                augmented[i] = np.roll(augmented[i], shift, axis=0)
    
    return augmented


# ============================================================================
# 3. CLASS BALANCING
# ============================================================================

def create_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create a weighted sampler for class-balanced training."""
    unique, counts = np.unique(labels, return_counts=True)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(labels),
        replacement=True
    )


def create_stratified_val_split(
    n_samples: int,
    labels: np.ndarray,
    val_fraction: float = 0.1,
    min_per_class: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Create stratified train/val split ensuring balanced validation set."""
    unique_classes = np.unique(labels)
    train_idx, val_idx = [], []
    
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        n_cls = len(cls_indices)
        
        # Ensure minimum samples per class in validation
        n_val = max(min_per_class, int(n_cls * val_fraction))
        n_val = min(n_val, n_cls - min_per_class)  # Keep min in train too
        
        # Shuffle and split
        np.random.shuffle(cls_indices)
        val_idx.extend(cls_indices[:n_val])
        train_idx.extend(cls_indices[n_val:])
    
    return np.array(train_idx), np.array(val_idx)


# ============================================================================
# 4. ENSEMBLE OF MODELS
# ============================================================================

class EnsembleModel(nn.Module):
    """Ensemble of multiple models with different initializations."""
    
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
    
    def forward(self, x, return_all: bool = False):
        """Forward pass - average predictions from all models."""
        outputs = [m(x, return_all=return_all) for m in self.models]
        
        if return_all:
            # Average each head's output
            return {
                "price": torch.stack([o["price"] for o in outputs]).mean(0),
                "trend": torch.stack([o["trend"] for o in outputs]).mean(0),
                "risk": torch.stack([o["risk"] for o in outputs]).mean(0),
                "state_logits": torch.stack([o["state_logits"] for o in outputs]).mean(0),
            }
        else:
            return torch.stack(outputs).mean(0)


def create_ensemble(cfg: Dict[str, Any], device: str, input_size: int, 
                    n_models: int = 3) -> List[nn.Module]:
    """Create multiple models with different seeds."""
    models = []
    model_cfg = cfg.get("model", {})
    
    for i in range(n_models):
        set_seed(42 + i * 100)  # Different seed for each model
        
        engine = UnifiedNeuralEngine({
            "device": device,
            "model": {
                "input_size": input_size,
                "hidden_size": int(model_cfg.get("hidden_size", 48)),
                "num_layers": int(model_cfg.get("num_layers", 3)),
                "dropout": float(model_cfg.get("dropout", 0.25)),
                "bidirectional": bool(model_cfg.get("bidirectional", False)),
                "state_classes": int(cfg.get("state_classes", 2)),
            },
        })
        engine.model.to(device)
        models.append(engine.model)
    
    return models


# ============================================================================
# ENHANCED TRAINING LOOP
# ============================================================================

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    price_loss_fn: nn.Module,
    state_loss_fn: nn.Module,
) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    model.eval()
    total_price = total_state = 0.0
    correct = total = 0
    n = 0
    
    with torch.no_grad():
        for xb, y_price, y_trend, y_risk, y_state in loader:
            xb = xb.to(device)
            y_price = y_price.to(device)
            y_state = y_state.to(device)
            
            out = model(xb, return_all=True)
            total_price += price_loss_fn(out["price"], y_price).item()
            total_state += state_loss_fn(out["state_logits"], y_state).item()
            
            preds = torch.argmax(out["state_logits"], dim=-1)
            correct += (preds == y_state).sum().item()
            total += y_state.numel()
            n += 1
    
    return {
        "price_loss": total_price / max(n, 1),
        "state_loss": total_state / max(n, 1),
        "state_acc": correct / max(total, 1),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer,
    weights: Dict[str, float],
    price_loss_fn: nn.Module,
    trend_loss_fn: nn.Module,
    risk_loss_fn: nn.Module,
    state_loss_fn: nn.Module,
    use_augmentation: bool = True,
) -> None:
    """Train one epoch with optional augmentation."""
    model.train()
    
    for xb, y_p, y_t, y_r, y_s in loader:
        # Apply augmentation
        if use_augmentation:
            xb_np = augment_sequences(xb.numpy())
            xb = torch.tensor(xb_np, dtype=torch.float32)
        
        xb = xb.to(device)
        y_p = y_p.to(device)
        y_t = y_t.to(device)
        y_r = y_r.to(device)
        y_s = y_s.to(device)
        
        out = model(xb, return_all=True)
        
        loss = (
            weights.get("price", 1.0) * price_loss_fn(out["price"], y_p)
            + weights.get("trend", 0.5) * trend_loss_fn(out["trend"], y_t)
            + weights.get("risk", 0.5) * risk_loss_fn(out["risk"], y_r)
            + weights.get("state", 7.0) * state_loss_fn(out["state_logits"], y_s)
        )
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def enhanced_train(
    config_path: str,
    csv_path: Optional[str] = None,
    n_candles: int = 3000,
    n_ensemble: int = 3,
    epochs: int = 100,
    patience: int = 20,
) -> Dict[str, Any]:
    """
    Enhanced training with all improvements:
    1. More data (n_candles)
    2. Data augmentation
    3. Class balancing (stratified split + weighted sampling)
    4. Ensemble of models
    """
    set_seed(42)
    cfg = load_config(config_path)
    device = cfg.get("device", "cpu")
    
    # Load data
    import pandas as pd
    if csv_path is None:
        data_dir = cfg.get("data", {}).get("data_dir") or "market_data"
        csv_files = sorted(Path(data_dir).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {data_dir}")
        csv_path = str(csv_files[0])
    
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Use more data if available
    if len(df) > n_candles:
        df = df.tail(n_candles)
    print(f"[Enhanced] Using {len(df)} candles")
    
    # Prepare sequences
    seq_len = cfg.get("sequence_length", 60)
    feature_cols = cfg.get("data", {}).get("required_features", ["open", "high", "low", "close", "volume"])
    
    x, y_price, meta = prepare_sequences(
        df=df,
        sequence_length=seq_len,
        target_column="close",
        feature_columns=list(feature_cols),
        target_shift=1,
        scale_features=True,
        scale_target=True,
    )
    
    targets = build_multitask_targets(
        df,
        sequence_length=seq_len,
        target_shift=1,
        close_col="close",
        risk_window=14,
        state_classes=int(cfg.get("state_classes", 2)),
    )
    
    # 3. STRATIFIED SPLIT for balanced validation
    train_idx, val_idx = create_stratified_val_split(
        len(x), targets.state, 
        val_fraction=0.1, 
        min_per_class=50
    )
    
    print(f"[Enhanced] Train: {len(train_idx)}, Val: {len(val_idx)}")
    
    # Print class distribution
    train_state = targets.state[train_idx]
    val_state = targets.state[val_idx]
    unique_train, counts_train = np.unique(train_state, return_counts=True)
    unique_val, counts_val = np.unique(val_state, return_counts=True)
    print(f"[Enhanced] Train distribution: {dict(zip(unique_train, counts_train))}")
    print(f"[Enhanced] Val distribution: {dict(zip(unique_val, counts_val))}")
    
    # Create datasets
    def make_tensors(idx):
        return (
            torch.tensor(x[idx], dtype=torch.float32),
            torch.tensor(y_price[idx], dtype=torch.float32).view(-1, 1),
            torch.tensor(targets.trend[idx], dtype=torch.float32).view(-1, 1),
            torch.tensor(targets.risk[idx], dtype=torch.float32).view(-1, 1),
            torch.tensor(targets.state[idx], dtype=torch.long),
        )
    
    train_data = TensorDataset(*make_tensors(train_idx))
    val_data = TensorDataset(*make_tensors(val_idx))
    
    # 3. BALANCED SAMPLER for training
    sampler = create_balanced_sampler(targets.state[train_idx])
    
    batch_size = cfg.get("batch_size", 64)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 4. CREATE ENSEMBLE
    print(f"[Enhanced] Creating ensemble of {n_ensemble} models")
    models = create_ensemble(cfg, device, input_size=x.shape[-1], n_models=n_ensemble)
    
    # Loss functions
    price_loss_fn = nn.HuberLoss(delta=1.0)
    trend_loss_fn = nn.MSELoss()
    risk_loss_fn = nn.BCELoss()
    state_loss_fn = nn.CrossEntropyLoss()
    
    weights = cfg.get("unified_head_loss_weights", {
        "price": 1.0, "trend": 0.5, "risk": 0.5, "state": 7.0
    })
    
    # Train each model in ensemble
    all_histories = []
    
    for model_idx, model in enumerate(models):
        print(f"\n{'='*50}")
        print(f"Training model {model_idx + 1}/{n_ensemble}")
        print(f"{'='*50}")
        
        set_seed(42 + model_idx * 100)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.get("learning_rate", 0.0005)),
            weight_decay=0.0,
        )
        
        best_acc = 0.0
        best_state = None
        no_improve = 0
        history = []
        
        for epoch in range(epochs):
            # 2. TRAIN WITH AUGMENTATION
            train_one_epoch(
                model, train_loader, device, optimizer, weights,
                price_loss_fn, trend_loss_fn, risk_loss_fn, state_loss_fn,
                use_augmentation=True
            )
            
            # Evaluate
            metrics = evaluate_model(model, val_loader, device, price_loss_fn, state_loss_fn)
            acc = metrics["state_acc"]
            
            marker = ""
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                marker = " ★"
            else:
                no_improve += 1
            
            history.append({"epoch": epoch + 1, "state_acc": acc})
            
            if (epoch + 1) % 5 == 0 or marker:
                print(f"  E{epoch+1:02d} | state_acc: {acc:.2%}{marker}")
            
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch + 1}")
                break
        
        # Restore best weights
        if best_state:
            model.load_state_dict(best_state)
        
        all_histories.append({"model": model_idx, "best_acc": best_acc, "history": history})
        print(f"  Model {model_idx + 1} best: {best_acc:.2%}")
    
    # Evaluate ensemble
    print(f"\n{'='*50}")
    print("Evaluating ensemble")
    print(f"{'='*50}")
    
    ensemble = EnsembleModel(models)
    ensemble_metrics = evaluate_model(ensemble, val_loader, device, price_loss_fn, state_loss_fn)
    ensemble_acc = ensemble_metrics["state_acc"]
    
    individual_accs = [h["best_acc"] for h in all_histories]
    avg_individual = sum(individual_accs) / len(individual_accs)
    
    print(f"\nIndividual model accuracies: {[f'{a:.2%}' for a in individual_accs]}")
    print(f"Average individual: {avg_individual:.2%}")
    print(f"Ensemble accuracy: {ensemble_acc:.2%}")
    
    # Save best model (or ensemble)
    model_dir = Path(cfg.get("MODEL_DIR", "trained_data/models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the best individual model
    best_model_idx = np.argmax(individual_accs)
    best_model = models[best_model_idx]
    
    # Prepare model params and data meta for checkpoint
    model_params = {
        "input_size": x.shape[-1],
        "hidden_size": int(cfg.get("model", {}).get("hidden_size", 48)),
        "num_layers": int(cfg.get("model", {}).get("num_layers", 3)),
        "dropout": float(cfg.get("model", {}).get("dropout", 0.25)),
        "bidirectional": False,
        "state_classes": int(cfg.get("state_classes", 2)),
    }
    
    data_meta = {
        "feature_columns": list(feature_cols),
        "target_column": "close",
        "sequence_length": seq_len,
        "target_shift": 1,
        "feature_scaler": meta.get("feature_scaler"),
        "target_scaler": meta.get("target_scaler"),
    }
    
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "config": cfg,
        "model_params": model_params,
        "data_meta": data_meta,
        "ensemble_accs": individual_accs,
        "ensemble_mean": avg_individual,
        "best_acc": individual_accs[best_model_idx],
    }, model_dir / "unified_market_net.pth")
    
    # Save all ensemble models
    for i, m in enumerate(models):
        torch.save({"model_state_dict": m.state_dict()}, model_dir / f"ensemble_model_{i}.pth")
    
    print(f"\nSaved best model (#{best_model_idx + 1}) to {model_dir / 'unified_market_net.pth'}")
    print(f"Saved all {n_ensemble} ensemble models")
    
    return {
        "best_individual_acc": max(individual_accs),
        "ensemble_acc": ensemble_acc,
        "individual_accs": individual_accs,
        "histories": all_histories,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced training with all improvements")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--csv", default=None, help="CSV data file")
    parser.add_argument("--candles", type=int, default=3000, help="Number of candles to use")
    parser.add_argument("--ensemble", type=int, default=3, help="Number of models in ensemble")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per model")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    args = parser.parse_args()
    
    result = enhanced_train(
        config_path=args.config,
        csv_path=args.csv,
        n_candles=args.candles,
        n_ensemble=args.ensemble,
        epochs=args.epochs,
        patience=args.patience,
    )
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Best individual model: {result['best_individual_acc']:.2%}")
    print(f"Ensemble accuracy: {result['ensemble_acc']:.2%}")
