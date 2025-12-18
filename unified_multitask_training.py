"""Unified multi-head training + evaluation.

This provides a minimal, runnable path to make the unified heads truly represent
*different tasks*, and to generate per-head metrics that can be summarized in a
"chat mode" via the reasoning engine.

Defaults are intentionally conservative and based on OHLCV CSVs.
"""

raise SystemExit(
    "Retired: Buddy training/inference/trading is only supported via main.py. "
    "Use: python main.py train-buddy"
)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
from datetime import datetime, timezone

import numpy as np

# PyTorch was removed from this repository. This legacy training path is not supported.
torch = None  # type: ignore


class _NN:
    class Module:  # pragma: no cover
        pass


nn = _NN()  # type: ignore


class DataLoader:  # pragma: no cover
    pass


class TensorDataset:  # pragma: no cover
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("TensorDataset is unavailable because PyTorch was removed.")

from data_processing import prepare_sequences
from feature_engineering import FeatureEngineering
from multitask_labels import build_multitask_targets, split_time_series
from neural_engine_unified import UnifiedNeuralEngine, safe_torch_load
from utils import load_config, setup_logging

logger = setup_logging(log_file="unified_multitask_training.log")


@dataclass
class HeadMetrics:
    price_loss: float
    trend_loss: float
    risk_loss: float
    state_loss: float
    state_acc: float


@dataclass
class LossFunctions:
    price: nn.Module
    trend: nn.Module
    risk: nn.Module
    state: nn.Module


def _to_loader(
    x: np.ndarray,
    y_price: np.ndarray,
    y_trend: np.ndarray,
    y_risk: np.ndarray,
    y_state: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y_price, dtype=torch.float32).view(-1, 1),
        torch.tensor(y_trend, dtype=torch.float32).view(-1, 1),
        torch.tensor(y_risk, dtype=torch.float32).view(-1, 1),
        torch.tensor(y_state, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    price_loss_fn: nn.Module,
    trend_loss_fn: nn.Module,
    risk_loss_fn: nn.Module,
    state_loss_fn: nn.Module,
) -> HeadMetrics:
    model.eval()

    total_price = total_trend = total_risk = total_state = 0.0
    correct_state = 0
    total_state_n = 0
    n_batches = 0

    with torch.no_grad():
        for xb, y_price, y_trend, y_risk, y_state in loader:
            xb = xb.to(device)
            y_price = y_price.to(device)
            y_trend = y_trend.to(device)
            y_risk = y_risk.to(device)
            y_state = y_state.to(device)

            out = model(xb, return_all=True)
            price = out["price"]
            trend = out["trend"]
            risk = out["risk"]
            state_logits = out["state_logits"]

            total_price += float(price_loss_fn(price, y_price).detach().cpu().item())
            total_trend += float(trend_loss_fn(trend, y_trend).detach().cpu().item())
            total_risk += float(risk_loss_fn(risk, y_risk).detach().cpu().item())
            total_state += float(state_loss_fn(state_logits, y_state).detach().cpu().item())

            preds = torch.argmax(state_logits, dim=-1)
            correct_state += int((preds == y_state).sum().detach().cpu().item())
            total_state_n += int(y_state.numel())

            n_batches += 1

    if n_batches == 0:
        return HeadMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    state_acc = float(correct_state / max(1, total_state_n))
    return HeadMetrics(
        price_loss=total_price / n_batches,
        trend_loss=total_trend / n_batches,
        risk_loss=total_risk / n_batches,
        state_loss=total_state / n_batches,
        state_acc=state_acc,
    )


def _resolve_csv_path(cfg: Dict[str, Any], csv_path: Optional[str]) -> str:
    if csv_path is not None:
        return csv_path

    data_dir = (
        cfg.get("data", {}).get("data_dir")
        or cfg.get("paths", {}).get("data_dir")
        or cfg.get("DATA_DIR")
        or "market_data"
    )
    candidates = sorted(Path(data_dir).glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No .csv files found under: {data_dir}")
    return str(candidates[0])


def _load_market_df(csv_path: str, *, add_all_features: bool = False, config: Optional[Dict[str, Any]] = None):
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Normalize columns to lowercase for label generation.
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

    if add_all_features:
        logger.info("Adding all engineered features (technical indicators, statistical, time, lag, rolling)...")
        fe = FeatureEngineering(config=config)
        df = fe.create_features(df, include_all=True)
        logger.info(f"Total features after engineering: {len(df.columns)}")

    return df


def _prepare_dataloaders(
    cfg: Dict[str, Any],
    df,
    *,
    seq_len: int,
    target_shift: int,
    feature_cols: list[str],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Any, DataLoader, DataLoader]:
    x, y_price, meta = prepare_sequences(
        df=df,
        sequence_length=seq_len,
        target_column="close",
        feature_columns=list(feature_cols),
        target_shift=target_shift,
        scale_features=bool(cfg.get("enable_data_scaling", True)),
        scale_target=bool(cfg.get("enable_data_scaling", True)),
    )

    targets = build_multitask_targets(
        df,
        sequence_length=seq_len,
        target_shift=target_shift,
        close_col="close",
        risk_window=int(cfg.get("risk_window", 14)),
        state_classes=int(cfg.get("state_classes", 3)),
    )

    if len(x) != len(y_price) or len(x) != len(targets.trend):
        raise RuntimeError("Target alignment mismatch")

    train_idx, val_idx = split_time_series(len(x), val_fraction=float(cfg.get("validation_split", 0.2)))

    # Log state label distribution for train and val sets
    train_state = targets.state[train_idx]
    val_state = targets.state[val_idx]
    unique_train, counts_train = np.unique(train_state, return_counts=True)
    unique_val, counts_val = np.unique(val_state, return_counts=True)
    logger.info("[State label distribution] Train: %s", dict(zip(unique_train, counts_train)))
    logger.info("[State label distribution] Val: %s", dict(zip(unique_val, counts_val)))

    train_loader = _to_loader(
        x[train_idx],
        y_price[train_idx],
        targets.trend[train_idx],
        targets.risk[train_idx],
        targets.state[train_idx],
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=True,
    )
    val_loader = _to_loader(
        x[val_idx],
        y_price[val_idx],
        targets.trend[val_idx],
        targets.risk[val_idx],
        targets.state[val_idx],
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=False,
    )

    return x, y_price, meta, targets, train_loader, val_loader


def _build_unified_engine(
    cfg: Dict[str, Any],
    *,
    device: str,
    input_size: int,
) -> UnifiedNeuralEngine:
    # Allow model config to be set via config.yaml (model: section or root)
    model_cfg = cfg.get("model", {})
    hidden_size = int(model_cfg.get("hidden_size", cfg.get("hidden_size", 64)))
    num_layers = int(model_cfg.get("num_layers", cfg.get("num_layers", 2)))
    dropout = float(model_cfg.get("dropout", cfg.get("dropout", 0.1)))
    bidirectional = bool(model_cfg.get("bidirectional", cfg.get("bidirectional", False)))
    state_classes = int(cfg.get("state_classes", model_cfg.get("state_classes", 3)))

    engine = UnifiedNeuralEngine(
        {
            "device": device,
            "model": {
                "input_size": int(input_size),
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "bidirectional": bidirectional,
                "state_classes": state_classes,
            },
        }
    )
    engine.model.to(device)
    return engine


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str,
    optimizer: torch.optim.Optimizer,
    weights: Dict[str, float],
    price_loss_fn: nn.Module,
    trend_loss_fn: nn.Module,
    risk_loss_fn: nn.Module,
    state_loss_fn: nn.Module,
    max_grad_norm: float,
) -> None:
    model.train()
    for xb, y_p, y_t, y_r, y_s in loader:
        xb = xb.to(device)
        y_p = y_p.to(device)
        y_t = y_t.to(device)
        y_r = y_r.to(device)
        y_s = y_s.to(device)

        out = model(xb, return_all=True)
        loss_price = price_loss_fn(out["price"], y_p)
        loss_trend = trend_loss_fn(out["trend"], y_t)
        loss_risk = risk_loss_fn(out["risk"], y_r)
        loss_state = state_loss_fn(out["state_logits"], y_s)

        loss = (
            float(weights.get("price", 1.0)) * loss_price
            + float(weights.get("trend", 0.5)) * loss_trend
            + float(weights.get("risk", 0.5)) * loss_risk
            + float(weights.get("state", 0.5)) * loss_state
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()


def _save_unified_model(model: nn.Module, cfg: Dict[str, Any], *, meta: Dict[str, Any]) -> str:
    model_dir = Path(cfg.get("MODEL_DIR") or cfg.get("paths", {}).get("model_dir") or "trained_data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "unified_market_net.pth"
    data_meta = {
        "feature_columns": list(meta.get("feature_columns") or []),
        "target_column": meta.get("target_column"),
        "sequence_length": int(meta.get("sequence_length") or 0),
        "target_shift": int(meta.get("target_shift") or 1),
        "feature_scaler": meta.get("feature_scaler"),
        "target_scaler": meta.get("target_scaler"),
    }

    model_cfg = cfg.get("model", {})
    model_params = {
        "input_size": int(model_cfg.get("input_size") or int(meta.get("feature_shape")[2] if meta.get("feature_shape") else 0) or 0),
        "hidden_size": int(cfg.get("hidden_size", model_cfg.get("hidden_size", 64))),
        "num_layers": int(cfg.get("num_layers", model_cfg.get("num_layers", 2))),
        "dropout": float(cfg.get("dropout", model_cfg.get("dropout", 0.1))),
        "bidirectional": bool(cfg.get("bidirectional", model_cfg.get("bidirectional", False))),
        "state_classes": int(cfg.get("state_classes", model_cfg.get("state_classes", 3))),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "model_params": model_params,
            "data_meta": data_meta,
        },
        out_path,
    )
    return str(out_path)


def _unified_model_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get("MODEL_DIR") or cfg.get("paths", {}).get("model_dir") or "trained_data/models")


def _maybe_resume_unified_weights(
    cfg: Dict[str, Any],
    model: nn.Module,
    *,
    device: str,
    resume_path: Optional[str],
) -> bool:
    """Best-effort resume of unified model weights.

    This intentionally resumes *weights only* (optimizer state is not persisted).
    """
    try:
        model_dir = _unified_model_dir(cfg)
        default_resume = model_dir / "unified_market_net.pth"
        chosen = Path(resume_path) if resume_path else default_resume
        if not (chosen.exists() and chosen.is_file()):
            return False

        # Use safe_torch_load which handles sklearn scaler allowlisting for PyTorch 2.6+
        payload = safe_torch_load(str(chosen), map_location=device)
        state = (payload or {}).get("model_state_dict") if isinstance(payload, dict) else None
        if not isinstance(state, dict) or not state:
            return False

        model.load_state_dict(state, strict=False)
        logger.info("Resumed unified weights from %s", str(chosen))
        return True
    except Exception as e:
        logger.warning("Failed to resume unified weights (%s). Training from scratch.", e)
        return False


def _unified_val_score(val_metrics: HeadMetrics, weights: Dict[str, float]) -> float:
    """Smaller is better."""
    try:
        return float(
            float(weights.get("price", 1.0)) * float(val_metrics.price_loss)
            + float(weights.get("trend", 0.5)) * float(val_metrics.trend_loss)
            + float(weights.get("risk", 0.5)) * float(val_metrics.risk_loss)
            + float(weights.get("state", 0.5)) * float(val_metrics.state_loss)
        )
    except Exception:
        return float(val_metrics.price_loss)


def _clone_state_dict(model: nn.Module) -> Dict[str, Any]:
    try:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    except Exception:
        return dict(model.state_dict())


def _adaptive_weight_adjustment(
    weights: Dict[str, float],
    val_metrics: "HeadMetrics",
    *,
    state_classes: int = 2,
    boost_factor: float = 1.15,
    max_weight: float = 5.0,
) -> Tuple[Dict[str, float], Optional[str]]:
    """Adjust weights to boost lagging heads. Returns (new_weights, adjustment_msg)."""
    losses = {
        "price": val_metrics.price_loss,
        "trend": val_metrics.trend_loss,
        "risk": val_metrics.risk_loss,
        "state": val_metrics.state_loss,
    }
    
    # Find median loss for comparison
    loss_vals = [v for v in losses.values() if v > 0]
    if not loss_vals:
        return weights, None
    median_loss = float(sorted(loss_vals)[len(loss_vals) // 2])
    
    # Check if state head is healthy by accuracy (don't boost if already good)
    # For binary: 65%+ is healthy (15% above 50% random baseline)
    # For 3-class: 48%+ is healthy (15% above 33% random baseline)
    state_baseline = 1.0 / max(state_classes, 2)
    state_healthy = val_metrics.state_acc >= (state_baseline + 0.15)
    
    new_weights = dict(weights)
    adjusted = None
    
    for head, loss in losses.items():
        # Skip state if accuracy is healthy (loss may be high but model is learning)
        if head == "state" and state_healthy:
            continue
        # Boost heads that are lagging (loss > 1.5x median) - more conservative threshold
        if loss > median_loss * 1.5 and new_weights.get(head, 0.5) < max_weight:
            old_w = new_weights.get(head, 0.5)
            new_weights[head] = min(old_w * boost_factor, max_weight)
            adjusted = f"↑ {head} weight: {old_w:.2f} → {new_weights[head]:.2f}"
    
    return new_weights, adjusted


def _train_select_best(
    cfg: Dict[str, Any],
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer,
    weights: Dict[str, float],
    loss_fns: LossFunctions,
    epochs: int,
    state_classes: int = 2,
    patience: int = 5,
    min_epochs: int = 10,
) -> tuple[list[Dict[str, Any]], Optional[Dict[str, Any]], Optional[float], Optional[int], Dict[str, float]]:
    """Train with adaptive weights and early stopping.
    
    Returns: (history, best_state, best_score, best_epoch, final_weights)
    """
    history: list[Dict[str, Any]] = []
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
    best_state: Optional[Dict[str, Any]] = None
    
    # Adaptive training state
    current_weights = dict(weights)
    no_improve_count = 0
    
    for epoch in range(1, int(epochs) + 1):
        _train_one_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            weights=current_weights,
            price_loss_fn=loss_fns.price,
            trend_loss_fn=loss_fns.trend,
            risk_loss_fn=loss_fns.risk,
            state_loss_fn=loss_fns.state,
            max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        )
        val_metrics = _evaluate(
            model,
            val_loader,
            device=device,
            price_loss_fn=loss_fns.price,
            trend_loss_fn=loss_fns.trend,
            risk_loss_fn=loss_fns.risk,
            state_loss_fn=loss_fns.state,
        )

        metrics_dict = {
            "epoch": epoch,
            "val": {
                "price_loss": val_metrics.price_loss,
                "trend_loss": val_metrics.trend_loss,
                "risk_loss": val_metrics.risk_loss,
                "state_loss": val_metrics.state_loss,
                "state_acc": val_metrics.state_acc,
            },
            "weights": dict(current_weights),
        }
        history.append(metrics_dict)

        # Compute score and check improvement
        score = _unified_val_score(val_metrics, current_weights)
        is_best = best_score is None or float(score) < float(best_score)
        
        if is_best:
            best_score = float(score)
            best_epoch = int(epoch)
            best_state = _clone_state_dict(model)
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Adaptive weight adjustment for lagging heads
        new_weights, adj_msg = _adaptive_weight_adjustment(
            current_weights, val_metrics, state_classes=state_classes
        )
        
        # Compact one-liner per epoch
        best_marker = " ★" if is_best else ""
        # Note: trend loss is tiny (e-7 to e-8) because FX returns are ~0.0001
        # Display with scientific notation for visibility
        trend_str = f"{val_metrics.trend_loss:.2e}" if val_metrics.trend_loss < 0.0001 else f"{val_metrics.trend_loss:.4f}"
        logger.info(
            "E%02d | P:%.4f T:%s R:%.3f S:%.3f (%.0f%%)%s",
            epoch,
            val_metrics.price_loss,
            trend_str,
            val_metrics.risk_loss,
            val_metrics.state_loss,
            val_metrics.state_acc * 100,
            best_marker,
        )
        
        # Log weight adjustment
        if adj_msg:
            logger.info("  %s", adj_msg)
            current_weights = new_weights

        # Early stopping check (only after min_epochs)
        if epoch >= min_epochs and no_improve_count >= patience:
            logger.info("⏹  Early stop: no improvement for %d epochs", patience)
            break

    return history, best_state, best_score, best_epoch, current_weights


def _default_metrics_path(cfg: Dict[str, Any]) -> Path:
    out_dir = Path(
        cfg.get("METRICS_DIR")
        or cfg.get("paths", {}).get("metrics_dir")
        or cfg.get("paths", {}).get("log_dir")
        or "trained_data/logs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "unified_training_metrics.json"


def _save_training_metrics(
    cfg: Dict[str, Any],
    *,
    csv_path: str,
    model_path: str,
    history: list[Dict[str, Any]],
) -> str:
    out_path = _default_metrics_path(cfg)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "csv": csv_path,
        "model_path": model_path,
        "history": history,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return str(out_path)


def _seed_everything(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _all_features_mode(cfg: Dict[str, Any], all_features: bool) -> bool:
    return bool(all_features) or bool(cfg.get("data", {}).get("all_features", False)) or bool(
        cfg.get("all_features", False)
    )


def _resolve_sequence_params(cfg: Dict[str, Any]) -> tuple[int, int]:
    seq_len = int(cfg.get("data", {}).get("sequence_length") or cfg.get("sequence_length") or 60)
    target_shift = int(cfg.get("data", {}).get("target_shift") or cfg.get("target_shift") or 1)
    return seq_len, target_shift


def _select_feature_columns(cfg: Dict[str, Any], df, *, all_features_mode: bool) -> list[str]:
    if all_features_mode:
        exclude_cols = {"close", "target", "label", "y"}
        feature_cols = [
            c for c in df.select_dtypes(include=["number"]).columns if c not in exclude_cols
        ]
        logger.info("All-features mode: using %d features for training", len(feature_cols))
        return feature_cols

    return cfg.get("data", {}).get("required_features") or ["open", "high", "low", "close", "volume"]


def _build_optimizer(cfg: Dict[str, Any], model: nn.Module, *, lr: float) -> torch.optim.Optimizer:
    weight_decay = float(cfg.get("weight_decay", cfg.get("training", {}).get("weight_decay", 0.01)))
    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=weight_decay)


def _resolve_yaml_config_path(config_path: str) -> str:
    return config_path if str(config_path).endswith(".yaml") else "config.yaml"


def _read_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _write_yaml(path: str, data: Dict[str, Any]) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _maybe_update_unified_weights_in_config(
    *,
    config_path: str,
    original_weights: Dict[str, float],
    final_weights: Dict[str, float],
) -> None:
    if final_weights == original_weights:
        return

    w_str = " ".join(f"{k}:{v:.2f}" for k, v in final_weights.items())
    logger.info("  📊 Recommended weights: %s", w_str)

    resolved_path = _resolve_yaml_config_path(config_path)
    try:
        config_data = _read_yaml(resolved_path)
        config_data["unified_head_loss_weights"] = {
            k: float(f"{v:.2f}") for k, v in final_weights.items()
        }
        _write_yaml(resolved_path, config_data)
        logger.info("  ✅ config.yaml updated with new unified_head_loss_weights.")
    except Exception as e:
        logger.warning("  ⚠️ Failed to update config.yaml automatically: %s", e)


def train_unified_multitask(
    config_path: str,
    *,
    csv_path: Optional[str] = None,
    resume_path: Optional[str] = None,
    seed: int = 42,
    all_features: bool = False,
) -> Dict[str, Any]:
    """Train the unified multi-head model and print per-head status via reasoning."""

    _seed_everything(seed)

    cfg = load_config(config_path)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    csv_path = _resolve_csv_path(cfg, csv_path)

    all_features_mode = _all_features_mode(cfg, all_features)

    df = _load_market_df(csv_path, add_all_features=all_features_mode, config=cfg)

    seq_len, target_shift = _resolve_sequence_params(cfg)
    feature_cols = _select_feature_columns(cfg, df, all_features_mode=all_features_mode)

    x, _, meta, _, train_loader, val_loader = _prepare_dataloaders(
        cfg,
        df,
        seq_len=seq_len,
        target_shift=target_shift,
        feature_cols=list(feature_cols),
    )

    engine = _build_unified_engine(cfg, device=device, input_size=int(x.shape[-1]))
    model = engine.model

    resumed = _maybe_resume_unified_weights(cfg, model, device=device, resume_path=resume_path)

    price_loss_fn = nn.HuberLoss(delta=float(cfg.get("huber_delta", 1.0)))
    trend_loss_fn = nn.MSELoss()
    risk_loss_fn = nn.BCELoss()
    state_loss_fn = nn.CrossEntropyLoss()

    state_classes = int(cfg.get("state_classes", cfg.get("model", {}).get("state_classes", 2)))

    weights = cfg.get("unified_head_loss_weights") or {
        "price": 1.0,
        "trend": 0.5,
        "risk": 0.5,
        "state": 0.5,
    }

    lr = float(cfg.get("learning_rate", cfg.get("training", {}).get("learning_rate", 1e-3)))
    optimizer = _build_optimizer(cfg, model, lr=lr)
    loss_fns = LossFunctions(
        price=price_loss_fn,
        trend=trend_loss_fn,
        risk=risk_loss_fn,
        state=state_loss_fn,
    )

    epochs = int(cfg.get("epochs", cfg.get("training", {}).get("epochs", 25)))
    patience = int(cfg.get("early_stop_patience", 5))
    min_epochs = int(cfg.get("min_epochs", 10))
    
    # Compact header
    logger.info("─" * 50)
    logger.info("Training ≤%d epochs | P=price T=trend R=risk S=state | ★=best", epochs)
    logger.info("─" * 50)
    
    history, best_state, best_score, best_epoch, final_weights = _train_select_best(
        cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        weights=weights,
        loss_fns=loss_fns,
        epochs=epochs,
        state_classes=state_classes,
        patience=patience,
        min_epochs=min_epochs,
    )
    if best_state is not None:
        try:
            model.load_state_dict(best_state, strict=False)
        except Exception:
            pass

    model_path = _save_unified_model(model, cfg, meta=meta)

    metrics_path = _save_training_metrics(
        cfg,
        csv_path=str(csv_path),
        model_path=str(model_path),
        history=history,
    )

    # Final summary
    if history:
        actual_epochs = len(history)
        logger.info("─" * 50)
        logger.info(
            "✓ Done | Best: E%02d (score=%.4f) | Ran %d/%d epochs",
            best_epoch or actual_epochs,
            best_score or 0,
            actual_epochs,
            epochs,
        )
        logger.info("  Model: %s", model_path)

        _maybe_update_unified_weights_in_config(
            config_path=config_path,
            original_weights=weights,
            final_weights=final_weights,
        )

    return {
        "csv": csv_path,
        "model_path": str(model_path),
        "metrics_path": metrics_path,
        "resumed": bool(resumed),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "final_weights": final_weights,
        "data_meta": {
            "feature_columns": list(meta.get("feature_columns") or []),
            "sequence_length": int(meta.get("sequence_length") or 0),
            "target_shift": int(meta.get("target_shift") or 1),
        },
        "history": history,
    }
