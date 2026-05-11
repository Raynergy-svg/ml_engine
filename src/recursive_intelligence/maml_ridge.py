"""MAML-based differentiable Ridge replacement (Tier 6 Phase 1 Prototype).

Replaces the Ridge confidence model with a small PyTorch neural net that supports
Model-Agnostic Meta-Learning (MAML) for fast regime adaptation.

Architecture:
    Input: 14 features (same as Ridge) → Linear(14, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 1) → Sigmoid * 100
    Output: confidence score 0-100 (same contract as Ridge)

MAML:
    - Inner loop: 3-5 gradient steps on regime-specific data (fast adaptation)
    - Outer loop: update meta-parameters so inner loop converges faster
    - Regime = task: each volatility regime is a separate "task" for MAML

Shadow mode:
    Runs alongside real Ridge. Both predict, only Ridge is used for trading.
    Logs comparison metrics for validation.

Integration:
    from src.recursive_intelligence.maml_ridge import MAMLRidge

    maml = MAMLRidge()
    # Shadow prediction (doesn't affect trading):
    maml_conf = maml.predict(features_14d)  # Returns 0-100

    # After trade closes, record outcome for meta-learning:
    maml.record_outcome(features, regime, outcome)

    # Periodic meta-train (every N outcomes):
    maml.meta_train()
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.scanner.automation.safe_json import safe_json_write

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("MAML Ridge: PyTorch not available — MAML disabled")

# Persistence
_STATE_DIR = Path("trained_data/models/maml_ridge")
_REGIME_NAMES = ["LOW", "NORMAL", "HIGH", "EXTREME"]


@dataclass
class MAMLConfig:
    """MAML hyperparameters."""
    # Architecture
    input_dim: int = 14
    hidden_dim: int = 64
    hidden_dim2: int = 32

    # MAML inner loop (per-regime adaptation)
    inner_lr: float = 0.1        # Stable with hardtanh + gradient clipping (verified via sweep)
    inner_steps: int = 5         # Number of gradient steps per task
    inner_grad_clip: float = 1.0  # Max gradient norm per inner step

    # MAML outer loop (meta-parameter update)
    outer_lr: float = 0.05      # Increased 50x: Reptile epsilon needs to be large enough to move
    meta_batch_size: int = 4    # Number of tasks (regimes) per outer step

    # Data management
    min_samples_per_regime: int = 10   # Min samples before including regime in meta-train
    max_buffer_per_regime: int = 200   # Max samples per regime (sliding window)
    meta_train_interval: int = 20      # Meta-train every N new outcomes

    # Shadow mode & auto-promotion
    shadow_mode: bool = True    # If True, predictions are logged but not used
    auto_promote: bool = True   # If True, automatically promote when benchmark says PROMOTE
    min_outcomes_for_promotion: int = 100  # Min trade outcomes before promotion is considered
    promotion_check_interval: int = 25    # Check promotion eligibility every N outcomes
    demotion_lookback: int = 50           # After promotion, check last N trades for regression
    demotion_mae_threshold: float = 5.0   # Demote if MAML MAE exceeds Ridge MAE by this much


if TORCH_AVAILABLE:
    class NeuralRidge(nn.Module):
        """Small differentiable network matching Ridge's 14→confidence contract."""

        def __init__(self, input_dim: int = 14, hidden_dim: int = 64, hidden_dim2: int = 32):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
            self.fc3 = nn.Linear(hidden_dim2, 1)
            self._init_weights()

        def _init_weights(self):
            """Kaiming init to prevent dead ReLUs at startup."""
            for m in [self.fc1, self.fc2]:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            # Output layer: small weights, bias=50 to center in valid range
            nn.init.normal_(self.fc3.weight, std=0.01)
            nn.init.constant_(self.fc3.bias, 50.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass. Returns confidence 0-100 (linear output, hardtanh-bounded)."""
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
            out = self.fc3(h).squeeze(-1)
            # hardtanh: gradient=1 inside [0,100], gradient=0 outside
            # Unlike sigmoid*100, this doesn't saturate WITHIN the valid range
            return F.hardtanh(out, 0.0, 100.0)

        def param_count(self) -> int:
            return sum(p.numel() for p in self.parameters())


class MAMLRidge:
    """MAML-based Ridge replacement with per-regime fast adaptation.

    Core idea:
        - Meta-parameters θ are shared across all regimes
        - For each regime, we do K inner-loop gradient steps to get θ'_regime
        - The outer loop updates θ so that θ'_regime performs well
        - At inference time, we adapt θ → θ'_current_regime in K steps
    """

    def __init__(self, config: Optional[MAMLConfig] = None):
        self.config = config or MAMLConfig()
        self._enabled = TORCH_AVAILABLE

        if not self._enabled:
            return

        # Meta-model (shared initialization)
        self._meta_model = NeuralRidge(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_dim2=self.config.hidden_dim2,
        )
        self._meta_optimizer = optim.Adam(
            self._meta_model.parameters(),
            lr=self.config.outer_lr,
        )

        # Per-regime adapted models (cached after inner loop)
        self._adapted_models: Dict[str, NeuralRidge] = {}

        # Per-regime data buffers: {regime → [(features, target), ...]}
        self._regime_buffers: Dict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)

        # Tracking
        self._outcomes_since_train: int = 0
        self._total_meta_trains: int = 0
        self._total_predictions: int = 0
        self._shadow_log: List[Dict[str, Any]] = []  # Last N shadow comparisons

        # Stats per regime
        self._regime_stats: Dict[str, Dict[str, float]] = {
            r: {"predictions": 0, "mae": 0.0, "adaptation_loss": 0.0}
            for r in _REGIME_NAMES
        }

        # Phase 2: Optional benchmark hook (set externally)
        self._benchmark: Optional[Any] = None

        # Promotion tracking
        self._promoted: bool = False           # True when MAML has been auto-promoted
        self._promoted_at_outcome: int = 0     # Outcome count when promoted
        self._promotion_history: List[Dict[str, Any]] = []  # Log of promotion/demotion events
        self._outcomes_since_promotion_check: int = 0

        # Load persisted state
        self._load_state()

        logger.info(
            "MAML Ridge initialized (params=%d, inner_lr=%.4f, inner_steps=%d, shadow=%s)",
            self._meta_model.param_count(),
            self.config.inner_lr,
            self.config.inner_steps,
            self.config.shadow_mode,
        )

    def predict(self, features: np.ndarray, regime: str = "NORMAL") -> float:
        """Predict confidence score 0-100, adapting to current regime.

        Uses the regime-adapted model if available, otherwise falls back to meta-model.

        Args:
            features: (14,) or (N, 14) numpy array
            regime: Current volatility regime name

        Returns:
            Confidence score 0-100
        """
        if not self._enabled:
            return 50.0  # Neutral fallback

        regime = regime.upper() if regime else "NORMAL"
        if regime not in _REGIME_NAMES:
            regime = "NORMAL"

        try:
            # Get the adapted model for this regime, or adapt on the fly
            model = self._get_adapted_model(regime)

            # Prepare input
            if features.ndim == 2:
                features = features[-1]  # Use last row
            x = torch.tensor(features.astype(np.float32)).unsqueeze(0)

            with torch.no_grad():
                confidence = float(model(x).item())

            confidence = max(0.0, min(100.0, confidence))
            self._total_predictions += 1
            self._regime_stats[regime]["predictions"] += 1

            return confidence

        except Exception as e:
            logger.debug("MAML Ridge predict failed: %s", e)
            return 50.0

    def predict_shadow(
        self,
        features: np.ndarray,
        regime: str,
        ridge_confidence: float,
    ) -> Dict[str, Any]:
        """Run shadow prediction alongside real Ridge. Returns comparison dict."""
        maml_conf = self.predict(features, regime)
        diff = maml_conf - ridge_confidence

        result = {
            "maml_confidence": round(maml_conf, 2),
            "ridge_confidence": round(ridge_confidence, 2),
            "diff": round(diff, 2),
            "abs_diff": round(abs(diff), 2),
            "regime": regime,
        }

        # Keep last 100 shadow comparisons
        self._shadow_log.append(result)
        if len(self._shadow_log) > 100:
            self._shadow_log = self._shadow_log[-100:]

        return result

    def record_outcome(
        self,
        features: np.ndarray,
        regime: str,
        target_confidence: float,
    ) -> None:
        """Record a trade outcome for future meta-training.

        Args:
            features: (14,) feature vector at trade entry time
            regime: Volatility regime at trade time
            target_confidence: The "true" confidence (derived from outcome).
                              High if trade won with good R:R, low if lost.
        """
        if not self._enabled:
            return

        regime = regime.upper() if regime else "NORMAL"
        if regime not in _REGIME_NAMES:
            regime = "NORMAL"

        if features.ndim == 2:
            features = features[-1]

        # Add to regime buffer
        self._regime_buffers[regime].append(
            (features.astype(np.float32).copy(), float(target_confidence))
        )

        # Sliding window
        max_buf = self.config.max_buffer_per_regime
        if len(self._regime_buffers[regime]) > max_buf:
            self._regime_buffers[regime] = self._regime_buffers[regime][-max_buf:]

        self._outcomes_since_train += 1
        self._outcomes_since_promotion_check += 1

        # Auto meta-train
        if self._outcomes_since_train >= self.config.meta_train_interval:
            self.meta_train()

        # Auto promotion check
        if (
            self.config.auto_promote
            and self._outcomes_since_promotion_check >= self.config.promotion_check_interval
        ):
            self._outcomes_since_promotion_check = 0
            result = self.check_promotion()
            if result.get("action") in ("promoted", "demoted"):
                logger.info("MAML promotion check result: %s", result.get("action"))

    def meta_train(self) -> Dict[str, Any]:
        """Run one round of MAML meta-training across available regimes.

        Outer loop: for each regime with enough data, do inner-loop adaptation
        then compute loss on held-out samples, backprop through the adaptation
        to update meta-parameters.

        Returns:
            Training stats dict
        """
        if not self._enabled:
            return {"status": "disabled"}

        # Find regimes with enough data
        eligible_regimes = [
            r for r in _REGIME_NAMES
            if len(self._regime_buffers[r]) >= self.config.min_samples_per_regime
        ]

        if len(eligible_regimes) < 1:
            return {"status": "insufficient_data", "regimes": {r: len(self._regime_buffers[r]) for r in _REGIME_NAMES}}

        total_outer_loss_val = 0.0
        regime_losses = {}
        regime_inner_losses: Dict[str, List[float]] = {}
        regime_support_losses: Dict[str, float] = {}
        regime_param_drift: Dict[str, float] = {}

        # FOMAML outer loop: Reptile-style meta-parameter update.
        # For each regime, adapt → compute query loss → update meta-params
        # toward adapted params that worked well on query data.
        for regime in eligible_regimes[:self.config.meta_batch_size]:
            buffer = self._regime_buffers[regime]

            # Split into support (inner loop) and query (outer loss)
            n = len(buffer)
            split = max(1, n // 2)
            support = buffer[:split]
            query = buffer[split:]

            if len(query) < 1:
                query = support[-2:]  # Fallback: use last 2 of support

            # === INNER LOOP: adapt meta-params to this regime ===
            adapted_params, inner_losses = self._inner_loop(support)

            # === OUTER LOSS: evaluate adapted params on query set ===
            with torch.no_grad():
                query_loss = self._compute_loss_with_params(adapted_params, query)
            regime_losses[regime] = float(query_loss.item())
            self._regime_stats[regime]["adaptation_loss"] = float(query_loss.item())
            total_outer_loss_val += float(query_loss.item())

            # Phase 2: Record inner-loop diagnostics per regime
            regime_inner_losses[regime] = inner_losses
            with torch.no_grad():
                _support_final_loss = inner_losses[-1] if inner_losses else 0.0
            regime_support_losses[regime] = _support_final_loss

            # === REPTILE-STYLE UPDATE: move meta-params toward adapted params ===
            # θ ← θ + ε * (θ'_adapted - θ)
            # Phase 2: Measure parameter drift magnitude before update
            with torch.no_grad():
                _param_drift = sum(
                    float((adapted_p.data - meta_p.data).norm().item())
                    for meta_p, adapted_p in zip(self._meta_model.parameters(), adapted_params)
                )
                regime_param_drift[regime] = _param_drift

                for meta_p, adapted_p in zip(self._meta_model.parameters(), adapted_params):
                    meta_p.data += self.config.outer_lr * (adapted_p.data - meta_p.data)

        outer_loss_val = total_outer_loss_val / max(len(eligible_regimes), 1)

        # Invalidate cached adapted models (meta-params changed)
        self._adapted_models.clear()

        self._outcomes_since_train = 0
        self._total_meta_trains += 1

        # Persist
        self._save_state()

        stats = {
            "status": "trained",
            "meta_trains": self._total_meta_trains,
            "regimes_used": eligible_regimes,
            "regime_losses": regime_losses,
            "outer_loss": outer_loss_val,
            # Phase 2 diagnostics
            "inner_losses": regime_inner_losses,
            "support_losses": regime_support_losses,
            "param_drift": regime_param_drift,
        }

        logger.info(
            "MAML Ridge meta-train #%d: outer_loss=%.4f, regimes=%s",
            self._total_meta_trains,
            outer_loss_val,
            {r: f"{l:.4f}" for r, l in regime_losses.items()},
        )

        # Phase 2: Feed benchmark if connected
        if self._benchmark is not None:
            try:
                self._benchmark.record_meta_train(stats)
            except Exception as _be:
                logger.debug("MAML benchmark record_meta_train error: %s", _be)

        return stats

    def _inner_loop(
        self,
        support_data: List[Tuple[np.ndarray, float]],
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """MAML inner loop: K gradient steps on support data.

        Returns:
            Tuple of (adapted_params, inner_losses) where inner_losses
            is the per-step loss trajectory for diagnostics.
        """
        # Start from meta-parameters (clone with grad tracking)
        adapted_params = [p.clone().requires_grad_(True) for p in self._meta_model.parameters()]

        # Build support tensors
        X = torch.tensor(np.array([s[0] for s in support_data], dtype=np.float32))
        Y = torch.tensor(np.array([s[1] for s in support_data], dtype=np.float32))

        inner_losses: List[float] = []

        for step in range(self.config.inner_steps):
            # Forward with current adapted params
            pred = self._forward_with_params(X, adapted_params)
            loss = F.smooth_l1_loss(pred, Y, beta=5.0)
            inner_losses.append(float(loss.item()))

            # First-Order MAML (FOMAML): drop second-order gradients for stability.
            grads = torch.autograd.grad(
                loss,
                adapted_params,
                create_graph=False,
                allow_unused=True,
            )

            # Gradient clipping to prevent single-step divergence
            grad_list = [g if g is not None else torch.zeros_like(p) for p, g in zip(adapted_params, grads)]
            total_norm = torch.sqrt(sum(g.norm() ** 2 for g in grad_list))
            clip_coef = self.config.inner_grad_clip / (total_norm + 1e-6)
            if clip_coef < 1.0:
                grad_list = [g * clip_coef for g in grad_list]

            # SGD update on adapted params
            adapted_params = [
                (p - self.config.inner_lr * g).detach().requires_grad_(True)
                for p, g in zip(adapted_params, grad_list)
            ]

        return adapted_params, inner_losses

    def _forward_with_params(
        self,
        x: torch.Tensor,
        params: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass using explicit parameters (not module state).

        This is the key to MAML: we need to do forward passes with
        intermediate adapted parameters while keeping the computation
        graph connected for second-order gradients.
        """
        # params layout: [fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias]
        h = F.relu(F.linear(x, params[0], params[1]))
        h = F.relu(F.linear(h, params[2], params[3]))
        out = F.linear(h, params[4], params[5]).squeeze(-1)
        return F.hardtanh(out, 0.0, 100.0)

    def _compute_loss_with_params(
        self,
        params: List[torch.Tensor],
        data: List[Tuple[np.ndarray, float]],
    ) -> torch.Tensor:
        """Compute Smooth L1 loss using explicit parameters on given data."""
        X = torch.tensor(np.array([d[0] for d in data], dtype=np.float32))
        Y = torch.tensor(np.array([d[1] for d in data], dtype=np.float32))
        pred = self._forward_with_params(X, params)
        return F.smooth_l1_loss(pred, Y, beta=5.0)

    def _get_adapted_model(self, regime: str) -> NeuralRidge:
        """Get a regime-adapted model. Caches adapted models between meta-trains."""
        if regime in self._adapted_models:
            return self._adapted_models[regime]

        # If we have support data for this regime, adapt
        if len(self._regime_buffers.get(regime, [])) >= 3:
            try:
                adapted_params, _inner_losses = self._inner_loop(self._regime_buffers[regime])

                # Create a new model with adapted parameters
                adapted = NeuralRidge(
                    input_dim=self.config.input_dim,
                    hidden_dim=self.config.hidden_dim,
                    hidden_dim2=self.config.hidden_dim2,
                )
                with torch.no_grad():
                    for p_adapted, p_model in zip(adapted_params, adapted.parameters()):
                        p_model.copy_(p_adapted.detach())

                self._adapted_models[regime] = adapted
                return adapted
            except Exception as e:
                logger.debug("MAML inner loop for %s failed: %s — using meta-model", regime, e)

        # Fallback: use meta-model directly
        return self._meta_model

    # ── Auto-Promotion / Demotion ────────────────────────────────────────

    def check_promotion(self) -> Dict[str, Any]:
        """Check if MAML should be promoted from shadow to production, or demoted back.

        Called automatically after every `promotion_check_interval` outcomes.
        Uses the benchmark's promotion verdict (which requires statistical significance,
        MAE advantage, regime divergence, etc.).

        Returns:
            Dict with action taken and reasoning.
        """
        if not self._enabled or self._benchmark is None:
            return {"action": "skip", "reason": "disabled_or_no_benchmark"}

        if not self.config.auto_promote:
            return {"action": "skip", "reason": "auto_promote_disabled"}

        # If already promoted, check for demotion instead
        if self._promoted:
            return self._check_demotion()

        # Not enough data yet
        total_outcomes = sum(len(b) for b in self._regime_buffers.values())
        if total_outcomes < self.config.min_outcomes_for_promotion:
            return {
                "action": "wait",
                "reason": f"need {self.config.min_outcomes_for_promotion} outcomes, have {total_outcomes}",
            }

        # Ask the benchmark for its verdict
        try:
            report = self._benchmark.generate_report()
        except Exception as e:
            logger.debug("Promotion check: benchmark report failed: %s", e)
            return {"action": "error", "reason": str(e)}

        verdict = report.get("promotion_verdict", {})
        decision = verdict.get("decision", "CONTINUE_SHADOW")

        if decision == "PROMOTE":
            return self._execute_promotion(verdict)
        else:
            return {
                "action": "continue_shadow",
                "decision": decision,
                "score": verdict.get("score", 0),
                "reasons": verdict.get("reasons", []),
            }

    def _execute_promotion(self, verdict: Dict[str, Any]) -> Dict[str, Any]:
        """Flip MAML from shadow to production."""
        self.config.shadow_mode = False
        self._promoted = True
        total_outcomes = sum(len(b) for b in self._regime_buffers.values())
        self._promoted_at_outcome = total_outcomes

        event = {
            "action": "promoted",
            "timestamp": __import__("time").time(),
            "outcome_count": total_outcomes,
            "meta_trains": self._total_meta_trains,
            "verdict_score": verdict.get("score", 0),
            "reasons": verdict.get("reasons", []),
        }
        self._promotion_history.append(event)
        self._save_state()

        logger.warning(
            "MAML Ridge AUTO-PROMOTED to production (score=%d, outcomes=%d, trains=%d)",
            verdict.get("score", 0),
            total_outcomes,
            self._total_meta_trains,
        )
        return event

    def _check_demotion(self) -> Dict[str, Any]:
        """After promotion, monitor for regression. Demote if MAML degrades."""
        if self._benchmark is None:
            return {"action": "skip", "reason": "no_benchmark"}

        # Need enough post-promotion trades to evaluate
        total_outcomes = sum(len(b) for b in self._regime_buffers.values())
        post_promotion_trades = total_outcomes - self._promoted_at_outcome
        if post_promotion_trades < self.config.demotion_lookback:
            return {
                "action": "monitoring",
                "post_promotion_trades": post_promotion_trades,
                "need": self.config.demotion_lookback,
            }

        # Check recent shadow log for regression
        # Use the benchmark's report which tracks running MAE
        try:
            report = self._benchmark.generate_report()
        except Exception as e:
            return {"action": "error", "reason": str(e)}

        mae_comp = report.get("mae_comparison", {})
        maml_mae = mae_comp.get("maml_mae")
        ridge_mae = mae_comp.get("ridge_mae")

        if maml_mae is not None and ridge_mae is not None:
            regression = maml_mae - ridge_mae
            if regression > self.config.demotion_mae_threshold:
                return self._execute_demotion(regression, report)

        return {
            "action": "promoted_ok",
            "post_promotion_trades": post_promotion_trades,
            "maml_mae": maml_mae,
            "ridge_mae": ridge_mae,
        }

    def _execute_demotion(self, regression: float, report: Dict[str, Any]) -> Dict[str, Any]:
        """Demote MAML back to shadow mode due to regression."""
        self.config.shadow_mode = True
        self._promoted = False
        total_outcomes = sum(len(b) for b in self._regime_buffers.values())

        event = {
            "action": "demoted",
            "timestamp": __import__("time").time(),
            "outcome_count": total_outcomes,
            "regression_mae": round(regression, 4),
            "reason": f"MAML MAE exceeded Ridge by {regression:.4f} (threshold: {self.config.demotion_mae_threshold})",
        }
        self._promotion_history.append(event)
        self._save_state()

        logger.warning(
            "MAML Ridge DEMOTED back to shadow (regression=%.4f, threshold=%.4f)",
            regression,
            self.config.demotion_mae_threshold,
        )
        return event

    @property
    def is_promoted(self) -> bool:
        """Whether MAML is currently promoted to production (not shadow)."""
        return self._promoted and not self.config.shadow_mode

    def _save_state(self) -> None:
        """Persist meta-model and buffers."""
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)

            # Save meta-model
            model_path = _STATE_DIR / "meta_model.pt"
            torch.save({
                "model_state_dict": self._meta_model.state_dict(),
                "optimizer_state_dict": self._meta_optimizer.state_dict(),
                "total_meta_trains": self._total_meta_trains,
                "total_predictions": self._total_predictions,
                "promoted": self._promoted,
                "promoted_at_outcome": self._promoted_at_outcome,
                "promotion_history": self._promotion_history[-20:],
                "shadow_mode": self.config.shadow_mode,
                "config": {
                    "input_dim": self.config.input_dim,
                    "hidden_dim": self.config.hidden_dim,
                    "hidden_dim2": self.config.hidden_dim2,
                    "inner_lr": self.config.inner_lr,
                    "inner_steps": self.config.inner_steps,
                    "outer_lr": self.config.outer_lr,
                },
            }, str(model_path))

            # Save regime buffers as numpy arrays
            buffer_path = _STATE_DIR / "regime_buffers.json"
            buffer_data = {}
            for regime, buf in self._regime_buffers.items():
                buffer_data[regime] = [
                    {"features": f.tolist(), "target": t}
                    for f, t in buf[-self.config.max_buffer_per_regime:]
                ]
            if not safe_json_write(buffer_path, buffer_data):
                logger.warning("MAML Ridge: buffer save failed")

            logger.debug("MAML Ridge: state saved (trains=%d)", self._total_meta_trains)
        except Exception as e:
            logger.warning("MAML Ridge: save_state failed: %s", e)

    def _load_state(self) -> None:
        """Load persisted state."""
        if not self._enabled:
            return

        model_path = _STATE_DIR / "meta_model.pt"
        if model_path.exists():
            try:
                checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
                self._meta_model.load_state_dict(checkpoint["model_state_dict"])
                self._meta_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self._total_meta_trains = checkpoint.get("total_meta_trains", 0)
                self._total_predictions = checkpoint.get("total_predictions", 0)
                self._promoted = checkpoint.get("promoted", False)
                self._promoted_at_outcome = checkpoint.get("promoted_at_outcome", 0)
                self._promotion_history = checkpoint.get("promotion_history", [])
                # Restore shadow_mode from persisted state (promotion survives restart)
                if self._promoted:
                    self.config.shadow_mode = checkpoint.get("shadow_mode", True)
                logger.info(
                    "MAML Ridge: loaded meta-model (trains=%d, preds=%d, promoted=%s)",
                    self._total_meta_trains, self._total_predictions, self._promoted,
                )
            except Exception as e:
                logger.warning("MAML Ridge: meta-model load failed: %s", e)

        buffer_path = _STATE_DIR / "regime_buffers.json"
        if buffer_path.exists():
            try:
                buffer_data = json.loads(buffer_path.read_text())
                for regime, entries in buffer_data.items():
                    self._regime_buffers[regime] = [
                        (np.array(e["features"], dtype=np.float32), float(e["target"]))
                        for e in entries
                    ]
                total = sum(len(b) for b in self._regime_buffers.values())
                logger.info("MAML Ridge: loaded %d buffered samples across %d regimes",
                           total, len(self._regime_buffers))
            except Exception as e:
                logger.warning("MAML Ridge: buffer load failed: %s", e)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "shadow_mode": self.config.shadow_mode,
            "promoted": self._promoted,
            "auto_promote": self.config.auto_promote,
            "total_meta_trains": self._total_meta_trains,
            "total_predictions": self._total_predictions,
            "outcomes_since_train": self._outcomes_since_train,
            "param_count": self._meta_model.param_count() if self._enabled else 0,
            "regime_buffers": {r: len(b) for r, b in self._regime_buffers.items()},
            "regime_stats": self._regime_stats,
            "shadow_mae": self._shadow_mae(),
            "promotion_history": self._promotion_history[-5:],  # Last 5 events
        }

    def _shadow_mae(self) -> Optional[float]:
        """Mean absolute error between MAML and Ridge predictions in shadow mode."""
        if not self._shadow_log:
            return None
        return round(
            sum(s["abs_diff"] for s in self._shadow_log) / len(self._shadow_log),
            2,
        )
