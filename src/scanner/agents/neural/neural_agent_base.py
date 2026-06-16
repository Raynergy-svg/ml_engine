"""
Base class for neural scanner agents.

CRIT-4 FIXES applied (2026-06-14):
  1. Loss changed from MSE -> binary_crossentropy for stronger gradients at extremes.
  2. Added PrioritizedReplayBuffer for experience replay with importance sampling.
  3. Online updates now use 5–10 epochs with validation split + early stopping.
  4. Target is binary correctness (voted correctly = 1.0, incorrectly = 0.0)
     instead of smoothed reward (0.125 / 0.875).
  5. Added `build_policy` override point for temporal conv / LSTM policies.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.scanner.agents._team import AgentDecisionContext, AgentVerdict, _clip01, _safe_float, _last_value

logger = logging.getLogger(__name__)


@dataclass
class NeuralAgentConfig:
    """Shared hyperparameters for all neural agents."""

    hidden_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2
    learning_rate: float = 1e-3
    # Inference
    score_threshold: float = 0.55
    block_threshold: float = 0.30
    # Training (CRIT-4)
    min_samples_for_update: int = 100
    max_replay_size: int = 2000
    online_epochs: int = 5
    online_batch_size: int = 32
    validation_split: float = 0.2
    early_stop_patience: int = 3
    # US-010: Elastic Weight Consolidation
    use_ewc: bool = False
    ewc_lambda: float = 1e-4


class PrioritizedReplayBuffer:
    """Experience replay with prioritized sampling based on TD error.

    Higher priority for:
      - Recent experiences (non-stationary market)
      - Experiences with high surprise (|prediction - outcome| large)
    """

    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self.buffer: List[Dict[str, Any]] = []
        self.priorities: List[float] = []

    def add(self, experience: Dict[str, Any], outcome: int, prediction: float) -> None:
        """Add an experience with priority = |prediction - outcome| + epsilon."""
        priority = abs(prediction - outcome) + 1e-6
        self.buffer.append(experience)
        self.priorities.append(priority)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch with replacement according to priorities."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=True, p=probs)
        batch = [self.buffer[i] for i in indices]
        X = np.array([b["features"] for b in batch], dtype=np.float32)
        y = np.array([b["target"] for b in batch], dtype=np.float32)
        weights = np.array([1.0 / (len(self.buffer) * probs[i]) for i in indices], dtype=np.float32)
        weights = weights / weights.max()  # normalize importance weights
        return X, y, weights

    def __len__(self) -> int:
        return len(self.buffer)


class NeuralAgentBase(ABC):
    """Abstract base for a learned specialist agent."""

    name: str = "neural_base"
    base_weight: float = 1.0

    def __init__(self, config: Optional[NeuralAgentConfig] = None):
        self.cfg = config or NeuralAgentConfig()
        self.policy: Optional[keras.Model] = None
        self._replay_buffer = PrioritizedReplayBuffer(max_size=self.cfg.max_replay_size)
        self._trade_history: List[Dict[str, Any]] = []
        self._update_count = 0
        # neural-11: only predict from a policy that was loaded or trained.
        # A freshly-built (random) policy must NOT drive passed/block_trade.
        self._trained: bool = False
        # US-010: EWC state
        self._fisher_info: Optional[Dict[str, np.ndarray]] = None
        self._old_weights: Optional[Dict[str, np.ndarray]] = None
        # bugs-10: compile the EWC loss exactly once. Recompiling on every
        # online update resets/duplicates Adam slot variables (TF drops
        # optimizer state) and re-instantiates the EarlyStopping metric
        # contract. The compiled penalty reads the tf.Variable holders below,
        # so it always uses the latest Fisher generation without recompiling.
        self._ewc_compiled: bool = False
        # bugs-10: per-variable (fisher, old_weight) tf.Variable holders. The
        # compiled EWC loss closes over these; _compute_fisher_info assigns them
        # in place each generation so the penalty tracks the latest checkpoint
        # without recompiling. Keyed by trainable-variable name.
        self._ewc_var_holders: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def extract_features(self, ctx: AgentDecisionContext) -> np.ndarray:
        """Turn context into a fixed-length float vector."""
        ...

    def build_policy(self, input_dim: int) -> keras.Model:
        """Default MLP policy.  Subclasses may override for CNN / LSTM / attention."""
        inputs = keras.Input(shape=(input_dim,), name="features")
        x = inputs
        for i, units in enumerate(self.cfg.hidden_units):
            x = keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
            x = keras.layers.Dropout(self.cfg.dropout, name=f"drop_{i}")(x)
        # CRIT-4 FIX: sigmoid output + binary_crossentropy for classification
        score = keras.layers.Dense(1, activation="sigmoid", name="score", dtype="float32")(x)
        model = keras.Model(inputs=inputs, outputs=score, name=f"policy_{self.name}")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.cfg.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, ctx: AgentDecisionContext) -> AgentVerdict:
        """Emit an AgentVerdict using the learned policy."""
        features = self.extract_features(ctx)
        if features is None or len(features) == 0:
            return self._fallback_verdict(ctx, "feature_extraction_failed")

        if self.policy is None:
            self.policy = self.build_policy(len(features))

        # neural-11: an untrained/randomly-initialized policy must not emit a
        # real verdict — its arbitrary score could veto a live trade once
        # shadow mode is disabled. Abstain (score=0.5, no block) until the
        # policy has been loaded or has completed at least one online update.
        if not self._trained:
            return self._fallback_verdict(ctx, "policy_untrained")

        score_raw = float(self.policy.predict(features[np.newaxis, ...], verbose=0)[0, 0])
        score = _clip01(score_raw)

        passed = score >= self.cfg.score_threshold
        block_trade = score < self.cfg.block_threshold
        confidence_delta = (score - 0.5) * 0.10

        if block_trade:
            reason = f"{self.name} blocks (score={score:.2f})"
            reason_code = f"{self.name}_block"
        elif passed:
            reason = f"{self.name} supports (score={score:.2f})"
            reason_code = f"{self.name}_support"
        else:
            reason = f"{self.name} neutral (score={score:.2f})"
            reason_code = f"{self.name}_neutral"

        # Cache experience for replay buffer
        self._replay_buffer.add(
            experience={
                "features": features.tolist(),
                "score": score,
                "passed": passed,
                "pair": getattr(ctx.analysis, "pair", "UNKNOWN"),
                "direction": getattr(ctx.analysis, "direction", "HOLD"),
                "timestamp": getattr(ctx.analysis, "scan_time", None),
                "target": 0.5,  # placeholder, updated by record_outcome
            },
            outcome=int(passed),  # placeholder
            prediction=score,
        )

        return AgentVerdict(
            name=self.name,
            score=score,
            passed=passed,
            weight=self.base_weight,
            reason=reason,
            reason_code=reason_code,
            confidence_delta=confidence_delta,
            block_trade=block_trade,
            metadata={
                "raw_score": score_raw,
                "features": features.tolist(),
                "passed": passed,
                "score": score,
            },
        )

    def _fallback_verdict(self, ctx: AgentDecisionContext, reason_code: str) -> AgentVerdict:
        return AgentVerdict(
            name=self.name,
            score=0.5,
            passed=False,
            weight=self.base_weight,
            reason=f"{self.name} fallback — {reason_code}",
            reason_code=reason_code,
            confidence_delta=0.0,
            block_trade=False,
            metadata={"error": reason_code},
        )

    # ------------------------------------------------------------------
    # Training (CRIT-4)
    # ------------------------------------------------------------------
    def record_outcome(self, verdict_metadata: Dict[str, Any], trade_won: bool) -> None:
        """Record the outcome of a trade for which this agent voted.

        CRIT-4 FIX: target is binary correctness (1.0 = voted correctly, 0.0 = wrong)
        instead of smoothed reward.  This is a proper classification label.
        """
        # neural-1: the agent's decision MUST be present. Defaulting a missing
        # 'passed' to False inverts every label (agent labeled 'correct' exactly
        # when the trade LOST), poisoning the policy. Skip the sample loudly
        # instead of training on a fabricated decision.
        if "passed" not in verdict_metadata:
            logger.error(
                "%s: record_outcome got verdict_metadata without 'passed' key "
                "(keys=%s); skipping sample to avoid inverted labels.",
                self.name, sorted(verdict_metadata.keys()),
            )
            return
        passed = bool(verdict_metadata["passed"])
        # Correctness: (voted FOR and won) OR (voted AGAINST and lost)
        target = 1.0 if (passed == trade_won) else 0.0
        features = verdict_metadata.get("features")
        if features is None:
            features = verdict_metadata.get("metadata", {}).get("features")
        if features is None:
            return

        self._trade_history.append({
            "target": target,
            "trade_won": trade_won,
            **verdict_metadata,
        })

        # Add to replay buffer with corrected target
        self._replay_buffer.add(
            experience={
                "features": features,
                "target": target,
                **verdict_metadata,
            },
            outcome=int(target),
            prediction=verdict_metadata.get("score", 0.5),
        )

        # Trigger online update when buffer is large enough
        if len(self._replay_buffer) >= self.cfg.min_samples_for_update:
            self._online_update()

    def _online_update(self) -> None:
        """CRIT-4 FIX: multi-epoch training with validation + early stopping.
        US-010: Adds Elastic Weight Consolidation (EWC) penalty when enabled.
        """
        if len(self._replay_buffer) < self.cfg.min_samples_for_update:
            return

        # Sample from prioritized replay buffer
        X, y, _ = self._replay_buffer.sample(len(self._replay_buffer))

        if len(X) < 20:
            return

        if self.policy is None:
            self.policy = self.build_policy(X.shape[1])

        # US-010: EWC penalty
        if self.cfg.use_ewc and self._fisher_info is not None and self._old_weights is not None:
            self._apply_ewc_penalty()

        callbacks = []
        if self.cfg.validation_split > 0 and len(X) >= 50:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.cfg.early_stop_patience,
                    restore_best_weights=True,
                    verbose=0,
                )
            )

        history = self.policy.fit(
            X, y,
            epochs=self.cfg.online_epochs,
            batch_size=self.cfg.online_batch_size,
            validation_split=self.cfg.validation_split if len(X) >= 50 else 0.0,
            callbacks=callbacks,
            verbose=0,
        )
        self._update_count += 1
        # neural-11: policy has now seen real outcomes — safe to predict from.
        self._trained = True
        final_loss = float(history.history["loss"][-1])
        val_loss = float(history.history.get("val_loss", [final_loss])[-1])
        logger.info(
            f"{self.name}: online update #{self._update_count} on {len(X)} samples "
            f"(loss={final_loss:.4f}, val_loss={val_loss:.4f})"
        )

        # US-010: Update Fisher Information after training
        if self.cfg.use_ewc:
            self._compute_fisher_info(X)

    # ------------------------------------------------------------------
    # US-010: Elastic Weight Consolidation
    # ------------------------------------------------------------------
    def _compute_fisher_info(self, X: np.ndarray) -> None:
        """Compute diagonal Fisher Information matrix from gradients."""
        if self.policy is None:
            return
        try:
            import tensorflow as tf
            fisher: Dict[str, np.ndarray] = {}
            old_weights: Dict[str, np.ndarray] = {}
            with tf.GradientTape() as tape:
                preds = self.policy(X, training=False)
                # bugs-9 / neural-13: the Fisher diagonal approximates the
                # sensitivity of the model's OWN log-likelihood at the trained
                # optimum. The previous code used tf.zeros_like(preds) as the
                # target, which systematically biases every gradient toward the
                # y=0 class and produces an invalid Fisher estimate (anchors
                # weights toward "predict 0 everywhere"). Use the model's own
                # predicted class y_hat = round(p) as the label — the standard
                # empirical-Fisher source — so the penalty protects the weights
                # the model actually relies on.
                y_hat = tf.stop_gradient(tf.round(preds))
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                    y_hat, preds
                ))
            grads = tape.gradient(loss, self.policy.trainable_variables)
            for var, grad in zip(self.policy.trainable_variables, grads):
                if grad is not None:
                    fisher[var.name] = np.square(grad.numpy())
                    old_weights[var.name] = var.numpy().copy()
            # Atomic publish: anchor (old_weights) and Fisher are written
            # together so the EWC penalty can never mix generations (bugs-10).
            self._fisher_info = fisher
            self._old_weights = old_weights
            # bugs-10: mirror into in-place tf.Variable holders so the
            # compile-once EWC loss always reads the freshest generation.
            if self._ewc_var_holders is None:
                self._ewc_var_holders = {}
            for name, f_arr in fisher.items():
                old_arr = old_weights[name]
                holder = self._ewc_var_holders.get(name)
                if holder is None:
                    self._ewc_var_holders[name] = (
                        tf.Variable(f_arr, dtype=tf.float32, trainable=False),
                        tf.Variable(old_arr, dtype=tf.float32, trainable=False),
                    )
                else:
                    holder[0].assign(f_arr)
                    holder[1].assign(old_arr)
            logger.debug("%s: Fisher info computed (%d vars)", self.name, len(fisher))
        except Exception as e:
            logger.warning("%s: Fisher computation failed: %s", self.name, e)

    def _apply_ewc_penalty(self) -> None:
        """Compile the EWC-regularized loss onto the policy exactly once.

        bugs-10: the previous implementation recompiled the policy on EVERY
        online update. Each ``compile()`` resets/duplicates the Adam slot
        variables (TF warns and silently drops optimizer state) and rebuilds the
        metric objects the EarlyStopping/val path depends on, so over many
        updates the optimizer state churned and early-stopping could be
        disabled. It also baked ``tf.constant`` snapshots of Fisher/old weights
        at trace time, so anchor and Fisher could drift apart across partial
        failures.

        Fix: compile once (guarded by ``_ewc_compiled``), preserving the single
        optimizer instance. The penalty reads ``tf.Variable`` holders updated in
        place each generation by ``_compute_fisher_info`` — so the one compiled
        closure always uses the latest Fisher/old-weight pair (written
        atomically), with no recompile.
        """
        if self.policy is None or self._fisher_info is None or self._old_weights is None:
            return
        if self._ewc_compiled:
            # Already compiled with the EWC loss; the in-place Variable holders
            # carry the freshest Fisher generation. Recompiling would clobber
            # optimizer state (the bug). Nothing to do.
            return
        try:
            import tensorflow as tf

            holders = self._ewc_var_holders or {}

            def ewc_loss(y_true, y_pred):
                base = tf.keras.losses.binary_crossentropy(y_true, y_pred)
                penalty = tf.constant(0.0, dtype=tf.float32)
                for var in self.policy.trainable_variables:
                    holder = holders.get(var.name)
                    if holder is None:
                        continue
                    f, old = holder
                    penalty += tf.reduce_sum(
                        tf.cast(f, var.dtype) * tf.square(var - tf.cast(old, var.dtype))
                    )
                return base + self.cfg.ewc_lambda * penalty

            self.policy.compile(
                optimizer=self.policy.optimizer,
                loss=ewc_loss,
                metrics=["accuracy"],
            )
            self._ewc_compiled = True
            logger.debug("%s: EWC penalty compiled (lambda=%.2e)", self.name, self.cfg.ewc_lambda)
        except Exception as e:
            logger.warning("%s: EWC penalty application failed: %s", self.name, e)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        if self.policy is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.policy.save(path)
        # Save replay buffer and trade history
        meta_path = Path(path).with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "update_count": self._update_count,
                "replay_buffer_size": len(self._replay_buffer),
                "trade_history_count": len(self._trade_history),
            }, f, indent=2)

    def load(self, path: str) -> bool:
        try:
            self.policy = keras.models.load_model(path)
            # neural-11: a loaded policy is a trained policy — safe to predict.
            self._trained = True
            logger.info(f"{self.name}: loaded policy from {path}")
            return True
        except Exception as e:
            logger.warning(f"{self.name}: failed to load policy from {path}: {e}")
            return False
