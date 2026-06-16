"""Unified training harness — dispatch to registered trainers by name.

Example:
    from src.training.harness import train
    train(model_type="sota", data_paths=["trained_data/harvest/EUR_USD_M15.parquet"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# Trainer registry -----------------------------------------------------------
_TRAINER_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a trainer factory under a model_type name."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        _TRAINER_REGISTRY[name] = fn
        logger.debug("Registered trainer '%s' under '%s'", fn.__name__, name)
        return fn
    return decorator


def list_trainers() -> List[str]:
    """Return all registered trainer names."""
    return list(_TRAINER_REGISTRY.keys())


def train(
    model_type: str,
    data_paths: List[str],
    *,
    save_dir: Optional[str] = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Dispatch training to the registered trainer for *model_type*.

    Args:
        model_type: Registered name (e.g. 'sota', 'neural').
        data_paths: Canonical OHLCV parquet/CSV files.
        save_dir: Where to write checkpoints.  Defaults vary by trainer.
        dry_run: If True, do not write artifacts.
        **kwargs: Trainer-specific hyperparameters.

    Returns:
        Dict with at least {'status': 'ok'|'error', 'model_type': ...}
    """
    if model_type not in _TRAINER_REGISTRY:
        return {
            "status": "error",
            "error": f"Unknown model_type '{model_type}'. Available: {list_trainers()}",
        }

    factory = _TRAINER_REGISTRY[model_type]
    resolved_paths = [str(Path(p).expanduser().resolve()) for p in data_paths if Path(p).exists()]
    if not resolved_paths:
        return {"status": "error", "error": "No valid data paths provided."}

    try:
        result = factory(data_paths=resolved_paths, save_dir=save_dir, dry_run=dry_run, **kwargs)
        return {"status": "ok", "model_type": model_type, "result": result}
    except Exception as exc:
        logger.exception("Trainer '%s' failed: %s", model_type, exc)
        return {"status": "error", "model_type": model_type, "error": str(exc)}


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------
@register("sota")
def _train_sota(
    data_paths: List[str],
    save_dir: Optional[str] = None,
    dry_run: bool = False,
    pretrain_only: bool = False,
    finetune_only: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    from src.sota_core import RawSequenceModel, ModelConfig, SOTATrainer, TrainerConfig

    cfg = TrainerConfig(**kwargs)
    model = RawSequenceModel(ModelConfig(**{k: v for k, v in kwargs.items() if k in {
        "seq_len", "d_model", "num_heads", "num_layers", "dropout", "learning_rate"}}))
    trainer = SOTATrainer(model=model, config=cfg)

    if dry_run:
        return {"message": "dry_run — would pretrain + finetune", "data_paths": data_paths}

    metrics: Dict[str, Any] = {}
    if not finetune_only:
        metrics["pretrain"] = trainer.pretrain(data_paths, save_dir=save_dir or "trained_data/models/sota_pretrain")
    if not pretrain_only:
        metrics["finetune"] = trainer.finetune(data_paths, save_dir=save_dir or "trained_data/models/sota_finetuned")
    return metrics


@register("neural")
def _train_neural(
    data_paths: List[str],
    save_dir: Optional[str] = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    from src.scanner.agents.neural.trainer import NeuralAgentTrainer

    if dry_run:
        return {"message": "dry_run — would train neural agents", "data_paths": data_paths}

    trainer = NeuralAgentTrainer(save_dir=save_dir or "trained_data/models/neural_agents")
    # NeuralAgentTrainer may not expose a generic train(data_paths) API;
    # fall back to batch_train_from_parquet if available.
    if hasattr(trainer, "batch_train_from_parquet"):
        stats = trainer.batch_train_from_parquet(data_paths, **kwargs)
        return {"stats": stats}
    return {"message": "NeuralAgentTrainer does not support generic data_paths yet; register a custom wrapper."}
