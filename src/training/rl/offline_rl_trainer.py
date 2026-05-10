"""Offline RL trainer for the Decision Transformer.

Phase 1 (this commit): contract stubs only. The training loop, walk-forward
CV, W&B logging, and meta-pipeline handoff are all Phase 2 work. Methods raise
``NotImplementedError`` so drift off the phase plan fails loudly.

Architectural reference: ``src/training/rl/position_sizer.py`` — the existing
PPO sizer demonstrates the artifact-shipping plumbing (lazy stable-baselines3
import, scaler.pkl persistence, runtime adapter). The DT trainer follows the
same shape: lazy torch import, ``decision_transformer.pt`` artifact under
``trained_data/models/``, runtime-loaded by the gate stack.

Per spec §6, the runtime safety filter is non-negotiable: any DT prediction
that would violate ``.claude/rules/trading.md`` (R:R < 1.2:1, trend-veto,
MR-veto, staleness-block) is *discarded*; the existing gate stack's verdict
applies. The DT may *refuse* (NO_TRADE) but never *override* hard rules.

Per ``.claude/rules/improvement.md`` No-Mock Rule: tests for this trainer use
real ``tmp_path`` fixtures; no ``unittest.mock`` anywhere in this stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from src.training.rl.decision_transformer import (
        DecisionTransformer,
        DecisionTransformerConfig,
    )
    from src.training.rl.trajectory_loader import (
        Trajectory,
        TrajectoryLoader,
    )

logger = logging.getLogger(__name__)

# Artifact path. Mirrors the PPO sizer convention from position_sizer.py.
DT_MODEL_PATH = Path("trained_data/models/decision_transformer.pt")
DT_CONFIG_PATH = Path("trained_data/models/decision_transformer_config.json")


# ---------------------------------------------------------------------------
# Trainer config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OfflineRLTrainerConfig:
    """Hyperparameters for the offline RL training loop.

    Attributes:
        learning_rate: AdamW learning rate. Spec §3 default for tiny DT.
        weight_decay: AdamW weight decay.
        batch_size: Number of windows per gradient step.
        n_epochs: Number of full passes over the windowized dataset.
        warmup_steps: Linear LR warmup steps.
        n_folds: Walk-forward CV folds (purged k-fold per López de Prado).
        embargo: Trade-step embargo between train/test fold boundaries.
        holdout_frac: Fraction of each trajectory's tail held out for the
            promotion-decision metric (spec §6 decision rule).
        promotion_lift_threshold_R: Minimum holdout R-multiple lift over the
            current gate stack required to ship. Spec §6: 0.10R.
        promotion_pvalue_threshold: Maximum bootstrap-paired p-value to ship.
            Spec §6: 0.05.
        wandb_project: W&B project name. ``None`` disables logging.
        gradient_clip_norm: Global L2 clip on gradients. None disables.
        device: ``"cpu" | "cuda" | "mps"``. Defaults to "cpu" so CI runs.
    """

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    n_epochs: int = 20
    warmup_steps: int = 200
    n_folds: int = 5
    embargo: int = 5
    holdout_frac: float = 0.2
    promotion_lift_threshold_R: float = 0.10
    promotion_pvalue_threshold: float = 0.05
    wandb_project: Optional[str] = "buddy-decision-transformer"
    gradient_clip_norm: Optional[float] = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0; got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {self.batch_size}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1; got {self.n_epochs}")
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2; got {self.n_folds}")
        if not 0.0 < self.holdout_frac < 1.0:
            raise ValueError(
                f"holdout_frac must be in (0, 1); got {self.holdout_frac}"
            )
        if self.promotion_lift_threshold_R < 0:
            raise ValueError(
                "promotion_lift_threshold_R must be >= 0; got "
                f"{self.promotion_lift_threshold_R}"
            )
        if not 0.0 < self.promotion_pvalue_threshold < 1.0:
            raise ValueError(
                f"promotion_pvalue_threshold must be in (0, 1); got "
                f"{self.promotion_pvalue_threshold}"
            )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class OfflineRLTrainer:
    """Walk-forward offline RL trainer for the Decision Transformer.

    Phase 1: stub. Construction validates wiring; ``train`` and ``evaluate``
    raise.

    Phase 2 contract:

      1. ``train()``:
         - Call ``loader.load()`` → trajectories.
         - For each of ``n_folds`` walk-forward folds:
             a. Split into train + val (chronological per pair, embargo).
             b. ``windowize(train)`` → torch DataLoader.
             c. Forward DT, cross-entropy on action logits at action-token
                positions, AdamW with linear warmup + cosine decay.
             d. Track val action-accuracy + expected-R per epoch in W&B.
         - Aggregate fold metrics; log to W&B summary.
         - Refit on train+val (excluding final holdout) for promotion eval.
         - Save artifact to ``DT_MODEL_PATH``.

      2. ``evaluate(holdout_trajectories)``:
         - Run DT.predict_action on each holdout step.
         - Apply ``_apply_runtime_safety_filter`` (rule veto).
         - Reconstruct equity curve; compute holdout-R, Sharpe, max-DD.
         - Compare to baseline (current gate stack) on the *same* holdout
           slice; bootstrap-paired test for significance.
         - Return a ``HoldoutEvalResult`` dataclass (Phase 2 will define).

      3. ``maybe_promote(eval_result)``:
         - If ``eval_result.lift_R >= promotion_lift_threshold_R`` AND
           ``eval_result.pvalue < promotion_pvalue_threshold``:
             - Hand off ChangePackage to MetaManager (shadow → canary → live).
         - Else: log shelf decision; journal still feeds Phase-4 fallback.

      4. ``_apply_runtime_safety_filter(prediction, gate_context)``:
         - The non-negotiable runtime guard. Spec §6.
         - If the predicted action would create a trade that violates any of
           {R:R < 1.2:1, trend-veto, MR-veto with disagree>0.25,
           staleness-block with age>7d}, return ACTION_NO_TRADE.
         - Else: return the DT prediction unchanged.
         - This filter MUST live in code (not in ``config_adjustments.json``)
           and MUST be called by both training-time evaluation and runtime
           inference. Bypassing this filter is a P0 bug.
    """

    def __init__(
        self,
        loader: "TrajectoryLoader",
        model_config: "DecisionTransformerConfig",
        trainer_config: OfflineRLTrainerConfig,
    ) -> None:
        self.loader = loader
        self.model_config = model_config
        self.trainer_config = trainer_config
        self._model: Optional["DecisionTransformer"] = None

    # ------------------------------------------------------------------
    # Phase 2 entry points
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run walk-forward CV + final-fold refit + artifact save.

        Raises:
            NotImplementedError: Phase 2 work.
        """
        raise NotImplementedError(
            "OfflineRLTrainer.train: Phase 2 work. "
            "See class docstring for the contract."
        )

    def evaluate(self, holdout_trajectories: List["Trajectory"]) -> dict:
        """Run holdout evaluation against the current gate-stack baseline.

        Returns:
            Dict with keys: ``lift_R``, ``pvalue``, ``dt_holdout_R``,
            ``baseline_holdout_R``, ``sharpe``, ``max_dd``, ``win_rate``.
            Phase 2 will tighten this into a typed ``HoldoutEvalResult``.

        Raises:
            NotImplementedError: Phase 2 work.
        """
        raise NotImplementedError(
            "OfflineRLTrainer.evaluate: Phase 2 work."
        )

    def maybe_promote(self, eval_result: dict) -> bool:
        """Decide promotion per spec §6 decision rule.

        Returns:
            True if a ChangePackage was handed to MetaManager (shadow →
            canary → live). False if shelved.

        Raises:
            NotImplementedError: Phase 2 work.
        """
        raise NotImplementedError(
            "OfflineRLTrainer.maybe_promote: Phase 2 work. "
            "Phase 2 wires through MetaManager.intake(...)."
        )

    def _apply_runtime_safety_filter(
        self, predicted_action: int, gate_context: dict
    ) -> int:
        """The non-negotiable runtime safety guard (spec §6).

        Phase 2 implementation MUST:
          - Read ``gate_context["rr_ratio"]`` and force NO_TRADE if < 1.2.
          - Read ``gate_context["trend_passed"]`` and force NO_TRADE if False
            for directional actions (LONG/SHORT).
          - Read ``gate_context["mr_passed"]`` AND ``gate_context["model_disagreement"]``
            and force NO_TRADE if MR fails AND disagreement > 0.25.
          - Read ``gate_context["max_component_age_days"]`` and
            ``gate_context["uncertainty_score"]`` and force NO_TRADE if
            age > 7 AND uncertainty > 0.35.
          - Log every veto with reason for the trade journal.

        Raises:
            NotImplementedError: Phase 2 work. Trading rule veto is critical
            path; must be unit-tested with real fixtures.
        """
        raise NotImplementedError(
            "OfflineRLTrainer._apply_runtime_safety_filter: Phase 2 work. "
            "See spec §6 catastrophic guard. Non-negotiable."
        )


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "DT_MODEL_PATH",
    "DT_CONFIG_PATH",
    "OfflineRLTrainerConfig",
    "OfflineRLTrainer",
]
