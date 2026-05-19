"""W&B training integration — Keras callback + sklearn metric logger.

One file that instruments ALL Buddy trainers for wandb.ai observability:

1. `WandBTrainingCallback` (Keras) — streams per-epoch metrics, logs model
   artifacts on training end. Plugs into model.fit(callbacks=[...]).

2. `wandb_log_sklearn_training()` — one-shot logger for non-epoch trainers
   (Ridge, XGBoost, LightGBM, Random Forest). Logs final metrics + model artifact.

3. `wandb_training_run()` — context manager that wraps an entire
   scheduled_retrain.py invocation as a single W&B Run with nested groups
   per model type and per pair.

All functions no-op when BUDDY_WANDB_REGISTRY_ENABLED is unset. Never crash
the training loop because of observability.

Env:
    BUDDY_WANDB_REGISTRY_ENABLED=1  — enables W&B logging
    BUDDY_WANDB_PROJECT             — project name (default: buddy-agents)
    BUDDY_WANDB_ENTITY              — org/user entity (default: logged-in user)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_ENV_FLAG = "BUDDY_WANDB_REGISTRY_ENABLED"
_DEFAULT_PROJECT = "buddy-agents"


def _enabled() -> bool:
    v = os.getenv(_ENV_FLAG, "")
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _get_wandb():
    """Lazy import — returns the wandb module or None."""
    if not _enabled():
        return None
    try:
        import wandb
        return wandb
    except ImportError:
        return None


def _get_keras_callback_base():
    """Return tf.keras.callbacks.Callback if importable, else `object`.

    Inheriting from the real Callback base gives us free no-op
    implementations of EVERY lifecycle method Keras 3 may add — past, present,
    and future. Falling back to `object` keeps this module importable in
    environments without TensorFlow (e.g. sklearn-only test runs).
    """
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.callbacks import Callback as _Callback  # type: ignore[attr-defined]
        return _Callback
    except Exception:  # ImportError, RuntimeError on partial installs, etc.
        return object


_KERAS_CALLBACK_BASE = _get_keras_callback_base()


# ─────────────────────────────────────────────────────────────────
# 1. Keras callback — for Transformer + TCN epoch training
# ─────────────────────────────────────────────────────────────────

class WandBTrainingCallback(_KERAS_CALLBACK_BASE):  # type: ignore[misc,valid-type]
    """Keras-compatible callback that logs epoch metrics to W&B.

    Usage:
        cb = WandBTrainingCallback(model_name="transformer_direction", pair="joint")
        model.fit(..., callbacks=[cb, ...other_callbacks])
        # On training end, logs best metrics + model file as artifact.

    Inherits from tf.keras.callbacks.Callback when available. This gives us
    no-op implementations of every lifecycle method (on_train_begin,
    on_test_batch_end, etc.) for free, so Keras 3's unconditional dispatch
    (`callback_list.py:237: callback.on_train_begin(logs)`) never raises
    AttributeError. The explicit hook stubs below remain as defensive
    documentation of which hooks intentionally no-op.

    The base falls back to `object` in TF-less environments — the explicit
    stubs are then load-bearing.
    """

    def __init__(
        self,
        model_name: str = "unknown",
        pair: str = "joint",
        model_save_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        # Initialize the Keras Callback base when present so Keras-internal
        # state (e.g. self.params, self.model) is wired up correctly. Safe
        # no-op when base is `object`.
        try:
            super().__init__()
        except Exception:
            pass
        self.model_name = model_name
        self.pair = pair
        self.model_save_path = model_save_path
        self._config = config or {}
        self._run = None
        self._wandb = _get_wandb()
        self._epoch_count = 0
        self._best_val_acc = 0.0
        self._best_epoch = 0
        self._start_time = time.time()
        self._model = None

        if self._wandb is not None:
            try:
                project = os.getenv("BUDDY_WANDB_PROJECT", _DEFAULT_PROJECT)
                entity = os.getenv("BUDDY_WANDB_ENTITY") or None
                run_name = f"train-{model_name}-{pair}-{datetime.now(timezone.utc).strftime('%H%M%S')}"
                self._run = self._wandb.init(
                    entity=entity,
                    project=project,
                    name=run_name,
                    job_type="training",
                    group=f"retrain-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                    tags=[model_name, pair, "keras"],
                    config={
                        "model_name": model_name,
                        "pair": pair,
                        **self._config,
                    },
                    reinit=True,
                )
                logger.info("W&B training run started: %s", run_name)
            except Exception as exc:
                logger.debug("W&B init failed for %s: %s", model_name, exc)
                self._wandb = None

    def set_params(self, params: Dict[str, Any]) -> None:
        """Called by Keras at training start."""
        if self._run and self._wandb:
            self._run.config.update({"keras_params": params})

    def set_model(self, model: Any) -> None:
        """Called by Keras — store model ref."""
        self._model = model

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras lifecycle hook; intentionally no-op beyond start-time reset."""
        self._start_time = time.time()

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras lifecycle hook; metrics are logged at epoch end."""
        return None

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras lifecycle hook; batch-level logging is intentionally disabled."""
        return None

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras lifecycle hook; batch-level logging is intentionally disabled."""
        return None

    def on_test_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras validation lifecycle hook; epoch-end metrics handle logging."""
        return None

    def on_test_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras validation lifecycle hook; epoch-end metrics handle logging."""
        return None

    def on_test_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras validation lifecycle hook; batch-level logging is disabled."""
        return None

    def on_test_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras validation lifecycle hook; batch-level logging is disabled."""
        return None

    def on_predict_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras prediction lifecycle hook; intentionally no-op."""
        return None

    def on_predict_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras prediction lifecycle hook; intentionally no-op."""
        return None

    def on_predict_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras prediction lifecycle hook; intentionally no-op."""
        return None

    def on_predict_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Keras prediction lifecycle hook; intentionally no-op."""
        return None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Stream epoch metrics to W&B."""
        if not self._wandb or not self._run:
            return
        logs = logs or {}
        self._epoch_count = epoch + 1
        try:
            metrics = {
                f"{self.model_name}/train_loss": logs.get("loss"),
                f"{self.model_name}/train_accuracy": logs.get("accuracy"),
                f"{self.model_name}/val_loss": logs.get("val_loss"),
                f"{self.model_name}/val_accuracy": logs.get("val_accuracy"),
                f"{self.model_name}/epoch": epoch + 1,
                f"{self.model_name}/learning_rate": logs.get("lr"),
            }
            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}
            self._wandb.log(metrics, step=epoch + 1)

            val_acc = logs.get("val_accuracy", 0.0)
            if val_acc and val_acc > self._best_val_acc:
                self._best_val_acc = val_acc
                self._best_epoch = epoch + 1
        except Exception as exc:
            logger.debug("W&B epoch log failed: %s", exc)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log summary + model artifact on training completion."""
        if not self._wandb or not self._run:
            return
        try:
            duration = time.time() - self._start_time
            self._run.summary.update({
                f"{self.model_name}/best_val_accuracy": self._best_val_acc,
                f"{self.model_name}/best_epoch": self._best_epoch,
                f"{self.model_name}/total_epochs": self._epoch_count,
                f"{self.model_name}/duration_seconds": round(duration, 1),
            })

            # Log model file as artifact if path provided
            if self.model_save_path and Path(self.model_save_path).exists():
                art = self._wandb.Artifact(
                    name=f"{self.model_name}-{self.pair}",
                    type="model",
                    metadata={
                        "best_val_accuracy": self._best_val_acc,
                        "best_epoch": self._best_epoch,
                        "total_epochs": self._epoch_count,
                    },
                )
                art.add_file(self.model_save_path)
                self._run.log_artifact(art)

            self._run.finish()
            logger.info(
                "W&B run finished: %s (best_val=%.4f @ epoch %d, %d epochs, %.1fs)",
                self.model_name, self._best_val_acc, self._best_epoch,
                self._epoch_count, duration,
            )
        except Exception as exc:
            logger.debug("W&B train_end failed: %s", exc)
            try:
                self._run.finish()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────
# 2. Sklearn / LightGBM / XGBoost — one-shot metric logger
# ─────────────────────────────────────────────────────────────────

def wandb_log_sklearn_training(
    model_name: str,
    pair: str,
    metrics: Dict[str, Any],
    model_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> None:
    """Log a non-epoch trainer's final metrics + artifact to W&B.

    Call after .train() completes and model is saved to disk. One W&B run
    per model-pair combination, grouped by retrain date.
    """
    wandb = _get_wandb()
    if wandb is None:
        return

    try:
        project = os.getenv("BUDDY_WANDB_PROJECT", _DEFAULT_PROJECT)
        entity = os.getenv("BUDDY_WANDB_ENTITY") or None
        run_name = f"train-{model_name}-{pair}-{datetime.now(timezone.utc).strftime('%H%M%S')}"

        with wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            job_type="training",
            group=f"retrain-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            tags=[model_name, pair, *(tags or [])],
            config={
                "model_name": model_name,
                "pair": pair,
                **(config or {}),
            },
            reinit=True,
        ) as run:
            # Prefix metrics with model name for dashboard grouping
            prefixed = {f"{model_name}/{k}": v for k, v in metrics.items()
                       if isinstance(v, (int, float, bool))}
            run.log(prefixed)
            run.summary.update(prefixed)

            if model_path and Path(model_path).exists():
                art = wandb.Artifact(
                    name=f"{model_name}-{pair}",
                    type="model",
                    metadata=metrics,
                )
                p = Path(model_path)
                if p.is_dir():
                    art.add_dir(str(p))
                else:
                    art.add_file(str(p))
                run.log_artifact(art)

        logger.info("W&B sklearn run logged: %s/%s metrics=%s", model_name, pair, metrics)
    except Exception as exc:
        logger.debug("wandb_log_sklearn_training failed for %s/%s: %s", model_name, pair, exc)


# ─────────────────────────────────────────────────────────────────
# 3. Top-level retrain run context manager
# ─────────────────────────────────────────────────────────────────

@contextmanager
def wandb_training_run(
    pairs: List[str],
    granularity: str = "H1",
    candles: int = 5000,
    reason: str = "scheduled",
) -> Iterator[Optional[Any]]:
    """Wrap an entire retrain invocation as a parent W&B Run.

    Individual model trainings (via callback or sklearn logger) create child
    runs in the same group. This parent run holds the retrain-level config:
    which pairs, why triggered, how many candles.

    Usage:
        with wandb_training_run(pairs, granularity, candles) as run:
            # ... do training ...
            if run:
                run.log({"overall_accuracy": 0.55})
    """
    wandb = _get_wandb()
    if wandb is None:
        yield None
        return

    try:
        project = os.getenv("BUDDY_WANDB_PROJECT", _DEFAULT_PROJECT)
        entity = os.getenv("BUDDY_WANDB_ENTITY") or None
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run = wandb.init(
            entity=entity,
            project=project,
            name=f"retrain-{ts}",
            job_type="retrain-orchestrator",
            group=f"retrain-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            tags=["retrain", reason, granularity],
            config={
                "pairs": pairs,
                "n_pairs": len(pairs),
                "granularity": granularity,
                "candles": candles,
                "reason": reason,
                "triggered_at": ts,
            },
            reinit=True,
        )
        logger.info("W&B retrain orchestrator run started: retrain-%s", ts)
        yield run
        run.finish()
    except Exception as exc:
        logger.debug("wandb_training_run context failed: %s", exc)
        yield None
